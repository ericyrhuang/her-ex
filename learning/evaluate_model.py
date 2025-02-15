import argparse
import io
import os
import torch
import json
import peano
import bootstrap
import torch.serialization
from transformers.generation.configuration_utils import CompileConfig



#################################################################################################################################################################################
#           Script to evaluate the model on the final goals. Make sure it is in the learning/ folder                                                                            #  
#           Usage: python learning/evaluate_model.py --model_path outputs/2024-11-12/23-22-47/ --final_goal_path "goals/nat-add-hard.json" --max_mcts_nodes=10000               # 
#           Alternative: python learning/evaluate_model.py --model_path outputs/2024-11-12/23-22-47/1.pt --final_goal_path "goals/nat-add-hard.json" --max_mcts_nodes=10000     # 
#################################################################################################################################################################################

def load_final_goals(path):
    goals_dict = json.load(open(path))
    final_goals = []
    solutions = []
    for goal in goals_dict["goals"]:
        final_goals.append(goal["theorem"])
        solutions.append(goal["solution"])

    return final_goals, solutions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to the run directory or model.pt file")
    parser.add_argument("--final_goal_path", help="Path to the final goal file")
    parser.add_argument("--max_mcts_nodes", help="Search budget of MCTS", default=10000)
    args = parser.parse_args()

    print("evaluating model" , args.model_path, "on", args.final_goal_path)

    theory_dict = {'name': 'nat-add', 'premises': ['eq_refl', 'eq_symm', 'rewrite', '+_z', '+_s', 'nat_ind']}
    with open(os.path.join(os.path.dirname(__file__), 'theories', theory_dict["name"] + '.p')) as f:
        theory = f.read()

    premises = theory_dict["premises"]
    d = peano.PyDerivation()
    d.incorporate(theory)
    
    # Verify that the model file exists
    if os.path.exists(args.model_path) and args.model_path.endswith(".pt"):
        models = {f"checkpoint_{os.path.basename(args.model_path).split('.')[0]}": torch.load(args.model_path, map_location=torch.device('cpu'))}
    elif os.path.exists(os.path.join(args.model_path, "model.pt")):
        models = {f"checkpoint_{os.path.basename(args.model_path).split('.')[0]}": torch.load(os.path.join(args.model_path, "model.pt"), map_location=torch.device('cpu'))}
    elif os.path.exists(os.path.join(args.model_path, "0.pt")):
        models = {}
        for i in range(15):
            if os.path.exists(os.path.join(args.model_path, f"{i}.pt")):
                models[f"checkpoint_{i}"] = torch.load(os.path.join(args.model_path, f"{i}.pt"), map_location=torch.device('cpu'))
            else:
                continue
    else:
        raise FileNotFoundError(f"model_path is neither file nor directory: {args.model_path}")
    # load final_goal from final_goal_path 
    final_goal_path = args.final_goal_path

    if os.path.exists(final_goal_path):
        if final_goal_path.endswith('.json'):
            final_goals_formatted, _ = load_final_goals(final_goal_path)
        else:
            raise ValueError(f"final_goal_path is not a JSON file: {final_goal_path}")
    else:
        raise FileNotFoundError(f"final_goal_path does not exist: {final_goal_path}")

    torch.serialization.add_safe_globals({'CompileConfig': CompileConfig})

    json_results = {}
    final_goal_name = os.path.basename(final_goal_path).split(".")[0]
    for i in models.keys():
        # Set the search budget
        models[i]._val_search_budget = int(args.max_mcts_nodes)
        # dump the model 
        buff = io.BytesIO()
        torch.save(models[i], buff)
        agent_dump = buff.getvalue()
        # Evaluate the model
        print(f"Goal: {final_goal_name} - Evaluating model {i}")
        val_loss, num_mcts_steps = bootstrap.get_val_loss(agent_dump, final_goals_formatted, theory, premises)
        print(f"Validation loss: {val_loss},\t Number of MCTS steps: {sum(num_mcts_steps)/len(num_mcts_steps)}")
        json_results[f"checkpoint_{i}"] = {"val_loss": val_loss, "num_mcts_steps": sum(num_mcts_steps)/len(num_mcts_steps)}

    print(f"Saving results to {os.path.join(args.model_path, f'{final_goal_name}_per_checkpoint_val_loss.json')}")
    print(json_results)
    if args.model_path.endswith(".pt"):
        with open(os.path.join(os.path.dirname(args.model_path), f"{final_goal_name}_per_checkpoint_val_loss.json"), "w") as f:
            json.dump(json_results, f)
    else: 
        with open(os.path.join(args.model_path, f"{final_goal_name}_per_checkpoint_val_loss.json"), "w") as f:
            json.dump(json_results, f)

if __name__ == "__main__":
    main()