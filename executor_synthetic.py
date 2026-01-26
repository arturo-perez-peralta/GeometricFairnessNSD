import argparse
import json
import os
import torch
import numpy as np
from run_filtering import run_experiment

NUM_FEATURES = 10
NUM_SAMPLES = 1000
GROUPS = 2
SEED = 42

np.random.seed(SEED)

def generate_base_sim_params():
    mu_base = np.zeros(NUM_FEATURES)
    mus = [mu_base.copy() for _ in range(GROUPS)]
    
    sigmas = [1.0 for _ in range(GROUPS)]
    
    weights = np.random.uniform(-1, 1, size=NUM_FEATURES + 1)
    
    return {
        "ns": [NUM_SAMPLES, NUM_SAMPLES],
        "mus": mus,
        "sigmas": sigmas,
        "psens": 0.5,
        "weights": weights,
        "threshold": 0.5,
        "seed": SEED
    }

SIMULATION_GRID = [
    {"gaps": [0.0, 0.0], "gap_name": "No_Bias"},
    {"gaps": [0.5, 0.5], "gap_name": "Medium_Bias"},
    {"gaps": [1.0, 1.0], "gap_name": "High_Bias"},
    {"gaps": [2.0, 2.0], "gap_name": "Extreme_Bias"}
]

MODELS_TO_RUN = [
    "polydia", "polybun", "polygen",
    "specdia", "specbun", "specgen",
    "vectdia", "vectbun", "vectgen"
]

COMMON_PARAMS = {
    "lr": 0.01,
    "weight_decay": 5e-4,
    "epochs": 200,
    "seed": SEED,
    "metric": "euclidean",
    "k": 10,
    "train_ratio": 0.5,
    "val_ratio": 0.25,
    "test_ratio": 0.25,
    "trace": False
}

SHEAF_PARAMS = {
    "layers": 2,
    "hidden_channels": 32,
    "input_dropout": 0.5,
    "dropout": 0.5,
    "d": 2,
    "sheaf_act": "tanh",
    "orth": "matrix_exp", 
    "left_weights": True,
    "right_weights": True,
    "sparse_learner": False,
    "use_act": True,
    "second_linear": False,
    "linear": True,
    "edge_weights": True,
    "add_lp": False,
    "add_hp": False
}

POLY_PARAMS = {
    "k_eig": 200,
    "k_select": 5,
    "K_cheb": 30,
    "gamma": 0.5,
    "sigma": 0.01
}

SPEC_PARAMS = {
    "k_eig": 200,
    "k_select": 10,
    "f": 0.5
}

VECT_PARAMS = {
    "reg_strength": 1.0
}

def get_config(model_str, sim_config):
    params = COMMON_PARAMS.copy()
    
    sim_kwargs = generate_base_sim_params()
    sim_kwargs['gap'] = sim_config['gaps'] 
    
    params.update({
        "name": "simul",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "distance": False,
        "subset": False,
        "knn": True,
        "star": False,
        "folds": 5,
        "delta": 0.,
        "kwargs": sim_kwargs 
    })

    model_args = SHEAF_PARAMS.copy()

    if 'dia' in model_str:
        model_args['normalised'] = True
        model_args['deg_normalised'] = False
    elif 'bun' in model_str:
        model_args['deg_normalised'] = False
        model_args['normalised'] = True
        model_args['orth'] = 'matrix_exp'
    elif 'gen' in model_str:
        model_args['normalised'] = True
        model_args['deg_normalised'] = False

    if 'poly' in model_str:
        model_args.update(POLY_PARAMS)
    elif 'spec' in model_str:
        model_args.update(SPEC_PARAMS)
    elif 'vect' in model_str:
        model_args.update(VECT_PARAMS)

    return params, model_args

def main():
    os.makedirs('results/simulation', exist_ok=True)
    
    for sim_cfg in SIMULATION_GRID:
        gap_name = sim_cfg['gap_name']
        
        for model_str in MODELS_TO_RUN:
            print(f"Running {model_str} on Scenario: {gap_name}")
            
            params, model_args = get_config(model_str, sim_cfg)
            
            try:
                results = run_experiment(
                    model_str=model_str,
                    model_args=model_args,
                    params=params,
                    compare=True
                )
                
                filename = f"results/simulation/{model_str}_{gap_name}.json"
                
                if 'args' in results:
                    if 'vectors' in results['args']:
                        del results['args']['vectors']
                    if 'kwargs' in results['args']:
                        if 'mus' in results['args']['kwargs']:
                            results['args']['kwargs']['mus'] = [x.tolist() for x in results['args']['kwargs']['mus']]
                        if 'weights' in results['args']['kwargs']:
                            results['args']['kwargs']['weights'] = results['args']['kwargs']['weights'].tolist()

                with open(filename, 'w') as f:
                    json.dump(results, f, indent=4)
                    
                if 'unfiltered' in results and 'filtered' in results:
                    acc_u = results['unfiltered']['test']['acc']
                    acc_f = results['filtered']['test']['acc']
                    print(f"  Result: {acc_u:.3f} -> {acc_f:.3f}")

            except Exception as e:
                print(f"  Failed: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()