#!/usr/bin/env python3

"""Implements the conjecture-prove bootstrapping learning loop."""

import asyncio
import os
import io
import json
import time
from datetime import timedelta
import random
import yaml

import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

import peano
import worker
from worker import StudentResult  # noqa
from hindsight import HindsightExample  # noqa
from util import format_blocks_with_indent, sample_batch, setup_mle_logger, value_color, save_json, load_final_goals
from conjecture import AgentLM, Context, sample_conjecture
from proofsearch import make_agent

from mle_logging import MLELogger

log = logging.getLogger(__name__)

FAIL = "fail"


DISTRIBUTED = os.environ.get('DISTRIBUTED', False)


def submit_task(agent_dump: bytes, theory: worker.BackgroundTheory, statement: str, search_budget=None):
    if DISTRIBUTED:
        return worker.try_prove.apply_async((agent_dump, theory, statement, search_budget))
    else:
        return worker.try_prove.run(agent_dump, theory, statement, search_budget)


def get_task_result(task):
    if DISTRIBUTED:
        return task.get()
    else:
        return task



async def teacher_loop(cfg: DictConfig, mle_log: MLELogger):
    log.info('Running in %s', 'distributed mode.' if DISTRIBUTED else 'single-process mode.')
    agent = make_agent(cfg, mle_log)

    # load goals from file and format them
    # FIXME(f.srambical): check whether the goal set is correctly formatted (check the first few finetuning examples)
    final_goals_formatted, _ = load_final_goals(os.path.join(os.path.dirname(__file__), '../goals', cfg.goals + '.json'))
    final_goals = ["Conj:(hard) " + g for g in final_goals_formatted]

    with open(os.path.join(os.path.dirname(__file__), 'theories', cfg.theory.name + '.p')) as f:
        theory = f.read()

    difficulty_buckets = sorted([list(cfg.difficulty_buckets[i].items())[0]
                                 for i in range(len(cfg.difficulty_buckets))],
                                key=lambda kv: kv[1])
    
    premises = cfg.theory.premises

    d = peano.PyDerivation()
    d.incorporate(theory)
    proven_conjectures = []
    seen_hindsight_goals = set()
    proofs = []
    student_results_final = []
    model_info = dict()

    continue_dir = cfg.get('continue')
    start_iteration = 0

    if continue_dir is not None:
        os.chdir(continue_dir)
        log.info('Continuing run from %s', continue_dir)
        log.info('Starting from iteration %d', start_iteration)
        # Find largest iteration number such that i.pt exists.
        if cfg.checkpoint_per_iteration:
            i = 0
            while os.path.exists(f'{i}.pt'):
                i += 1
            start_iteration = i
            agent = torch.load(f'{i}.pt')
            print('Loaded agent from', f'{i}.pt')
        else:
            agent = torch.load(f'model.pt')
            with open('model_info.yaml') as f:
                model_info = yaml.safe_load(f)
            start_iteration = model_info['iteration'] + 1

    if cfg.get('freeze_conjecturer', False):
        log.info('Ablation: Freezing conjecturer.')


    with open('log.jsonl', 'w') as log_file:
        elapsed_time = 0
        for i in range(start_iteration, cfg.agent.policy.total_iterations):

            
            conjectures = final_goals_formatted
            log.info('Skipping conjecture generation, using %d final goals', len(conjectures))

            log_file.write(json.dumps({'iteration': i,
                                  'msg': f'It #{i}: posing {len(conjectures)} conjectures.',
                                  'conjectures': conjectures}))
            log_file.write('\n')
            log_file.flush()

            # 2- Try to prove each of the conjectures
            examples = []
            begin_iter_time = time.perf_counter()
            student_results = prove_conjectures(agent, conjectures, theory, premises)
            if cfg.mcts_only:
                end_iter_time = time.perf_counter()
                elapsed_time += end_iter_time - begin_iter_time
                log.info(f'Time elapsed after iteration {i}: {timedelta(seconds=elapsed_time)}')
                break

            # 3- Train model on proofs and outcome of conjectures (easy, hard, timeout)
            # 3a- Look at all the success logprobs and compute the easy/hard threhsold.
            success_logprobs = get_log_probs(student_results)
            ratio_proven = len(success_logprobs)/len(conjectures)
            log.info('%d out of %d conjectures proven. ratio = %f', 
                        len(success_logprobs), len(conjectures), ratio_proven)

            # Add output of proving final goals to the list of proven conjectures
            student_results.extend(student_results_final)

            mean_hard_sol_log_prob = np.mean(success_logprobs) if success_logprobs else 0
            # 3b- Classify problems into easy/hard.
            proven_conjectures_iteration = []
            for student_result in student_results:
                # Outcome is the name of the first difficulty bucket that is larger than the logprob.
                if student_result.success:
                    outcome = 'hard'
                else:
                    outcome = FAIL

                if not cfg.get('freeze_conjecturer', False):
                    examples.append(f'Conj:({outcome}) ' + d.elaborate(student_result.problem))

                if student_result.success:
                    proven_conjectures_iteration.append(student_result.problem)
                    proven_conjectures.append(student_result.problem)
                    proofs.append(student_result.proof)

                examples.extend(student_result.extracted_examples)

            if cfg.train_policy_on_hindsight_examples:
                seen_hindsight_goals = set()
                hindsight_log_probs = []
                for h in student_result.hindsight_examples:
                    if h.goal not in seen_hindsight_goals:
                        outcome = 'hard'

                        if not cfg.get('freeze_conjecturer', False):
                            examples.append(f'Conj:({outcome}) ' + d.elaborate(student_result.problem))
                        examples.extend(h.examples)
                        seen_hindsight_goals.add(h.goal)
                        hindsight_log_probs.append(h.logprob)

                thresholds = [np.percentile(hindsight_log_probs, p)
                              for _, p in difficulty_buckets]
                hard_sol_log_probs = [logprob for logprob in hindsight_log_probs if logprob >= thresholds[0]]
                mean_hard_sol_log_prob = np.mean(hard_sol_log_probs) if hard_sol_log_probs else 0
                
            log_file.write(json.dumps({'iteration': i,
                                  'msg': f'Training on {len(examples)} examples.'}))
            log_file.write('\n')

            # 3c- Train model on conjecturing and proof search examples.
            log.info(f"{len(examples)} accumulated training examples.")
            agent.train(examples=examples, final_goals=final_goals, ratio_proven=ratio_proven, mle_log=mle_log)
            end_iter_time = time.perf_counter()
            val_loss, num_mcts_steps = get_val_loss(agent, final_goals_formatted, theory, premises)
            # FIXME(m.mahajan): This should not be hardcoded
            if val_loss != 10:
                log.info(f'Found solution during validation loss computation! Time elapsed: {timedelta(seconds=elapsed_time + time.perf_counter() - begin_iter_time)}')

            elapsed_time += end_iter_time - begin_iter_time
            log.info(f'Time elapsed after iteration {i}: {timedelta(seconds=elapsed_time)}')
            log.info('Validation loss: %f', val_loss)
            log.info('Number of MCTS steps to solve final goals: %s', num_mcts_steps)

            final_goals_proven = [s for s in num_mcts_steps if s <= cfg.agent.max_mcts_nodes]
            log.info('Final goals proven: %d out of %d', len(final_goals_proven), len(final_goals))

            mle_log.update({'num_iterations': i},
                        {'val_loss': val_loss,
                        'final_goals_proven': len(final_goals_proven),
                        'ratio_proven': ratio_proven,
                        'mean_hard_sol_log_probs': mean_hard_sol_log_prob})


            mle_log.save()

            save_json(examples, f'examples_{i}.json')
            save_json(proven_conjectures_iteration, f'proven_conj_{i}.json')
            if cfg.checkpoint_per_iteration:
                torch.save(agent, f"{i}.pt")
            else:
                torch.save(agent, f"model.pt")
                model_info['iteration'] = i
                with open('model_info.yaml', 'w') as f:
                    yaml.dump(model_info, f)

            # terminate the learning loop if all final goals are proven
            if len(final_goals_proven) == len(final_goals):
                log.info('All final goals proven')
                if cfg.early_exit:
                    break

def get_val_loss(agent, final_goals_formatted, theory, premises):
    # get logprobs of proving the final goals (with far more mcts steps)
    student_results_final = prove_conjectures(agent, final_goals_formatted, theory, premises, is_eval=True)
    success_logprobs_final = get_log_probs(student_results_final)

    if len(success_logprobs_final) > 0:
        mean_success_logprobs_final = sum(success_logprobs_final)/len(success_logprobs_final)
    else:
        mean_success_logprobs_final = -10

    num_mcts_steps = [s.iterations for s in student_results_final]
    return -mean_success_logprobs_final, num_mcts_steps


def prove_conjectures(agent, conjectures, theory, premises, is_eval=False):

    # Dump current agent.
    buff = io.BytesIO()
    torch.save(agent, buff)
    agent_dump = buff.getvalue()
    tasks = []
    log.info('Submitting tasks...')
    for conjecture in conjectures:
        tasks.append(submit_task(
            agent_dump,
            worker.BackgroundTheory(theory, premises),
            conjecture, 
            is_eval))

    student_results = []

    log.info('Collecting %d results from workers.', len(tasks))

    for task in tasks:
        student_result = get_task_result(task)

        if student_result.error:
            log.error('Error in prover process!')
            log.error(student_result.error)
            continue

        student_results.append(student_result)
    return student_results


def get_log_probs(student_results):

    success_logprobs = []

    for student_result in student_results:
        if student_result.success:
            success_logprobs.append(student_result.logprob)

    return success_logprobs



@hydra.main(version_base="1.2", config_path="config", config_name="bootstrap")
def main(cfg: DictConfig):
    log.info('Running from: %s', os.getcwd())
    
    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    mle_log = setup_mle_logger(cfg)

    if cfg.task == 'teacher':
        asyncio.run(teacher_loop(cfg, mle_log))

if __name__ == '__main__':
    import sys
    original_argv = sys.argv
    sys.argv = [original_argv[0]]
    try:
        main()
    finally:
        sys.argv = original_argv
