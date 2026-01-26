import argparse
import json
import os
import torch
import numpy as np
from run_filtering import run_experiment

DATASET_NAME = "pokec_z"
MODELS_TO_RUN = [
    "vectdia",
    "polydia",
    "specdia",
]


GRID_F = [0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 20., 21., 22., 23., 24., 31., 32., 33., 34., 35., 36., 37., 38., 39.]
GRID_GAMMA = [0., 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.,  1.1, 3.]
GRID_REG = [ 0.,   0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.,   1.1,  1.3,  1.4, 1.5,  1.6,  1.7,  1.8,  2.,   2.5, 22.,  28.,  30.,  39.2, 39.6, 40.,  53.,  58.]


COMMON_PARAMS = {
    "lr": 3e-3,
    "weight_decay": 1e-4,
    'beta1': 0.9,
    'beta2': 0.9999,
    "epochs": 20,
    "seed": 42,
    "metric": "euclidean",
    "k": 10,
    "train_ratio": 0.85,
    "val_ratio": 0.0,
    "test_ratio": 0.15,
    "trace": True,
    "kfold": False,
    "folds": 5,
    "early_stop": True,
    "patience": 20,
}



SHEAF_PARAMS = {
    'hidden_channels': 8,
    'output_dim': 1,
    'd': 3,
    'layers': 2,
    'max_t': 1.0,
    'linear': True,
    'use_act': True,
    'sheaf_act': 'elu',
    'orth': 'euler',
    'sparse_learner': True,
    'edge_weights': True,
    'normalised': True,
    'deg_normalised': False,
    'left_weights': False,
    'right_weights': False,
    'second_linear': False,
    'reg_strength': 0.5,
    'dropout': 0.5,
    'input_dropout': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'add_lp': False,
    'add_hp': False
}

POLY_PARAMS = {
    "k_eig": 100,
    "k_select": 3,
    "K_cheb": 71,
    "sigma": 0.0001,
    'left_weights': True,
    'right_weights': True,
    'second_linear': True
}

SPEC_PARAMS = {
    "k_eig": 200,
    "k_select": 3,
    'left_weights': True,
    'right_weights': True,
    'second_linear': True
}

VECT_PARAMS = {
    'left_weights': True,
    'right_weights': True,
    'second_linear': True
}

def get_config(model_str, dataset):
    params = COMMON_PARAMS.copy()
    params.update({
        "name": dataset,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "distance": False,
        "subset": False,
        "knn": False,
        "star": False,
        "folds": 5,
        "delta": 0.5,
        "kwargs": {'k': params['k'], 'metric': params['metric']}
    })

    model_args = SHEAF_PARAMS.copy()


    model_args['deg_normalised'] = False
    model_args['normalised'] = True

    if 'poly' in model_str:
        model_args.update(POLY_PARAMS)
    elif 'spec' in model_str:
        model_args.update(SPEC_PARAMS)
    elif 'vect' in model_str:
        model_args.update(VECT_PARAMS)

    return params, model_args

def main():
    print(f"Starting Grid Execution for Dataset: {DATASET_NAME}")
    os.makedirs('results/filter/tradeoff', exist_ok=True)

    for model_str in MODELS_TO_RUN:
        
        param_name = None
        grid_values = []

        if 'poly' in model_str:
            param_name = 'gamma'
            grid_values = GRID_GAMMA
        elif 'spec' in model_str:
            param_name = 'f'
            grid_values = GRID_F
        elif 'vect' in model_str:
            param_name = 'reg_strength'
            grid_values = GRID_REG
        
        for val in grid_values:
            print(f"\n{'-'*40}")
            print(f"Running: {model_str} | {param_name}: {val}")
            print(f"{'-'*40}")

            params, model_args = get_config(model_str, DATASET_NAME)
            model_args[param_name] = val
            print(model_args)
            print(param_name)
            print(val)

            try:
                results = run_experiment(
                    model_str=model_str,
                    model_args=model_args,
                    params=params,
                    compare=False 
                )

                filename = f"results/filter/tradeoff/{model_str}_{DATASET_NAME}_{param_name}_{val}.json"
                
                if 'args' in results and 'vectors' in results['args']:
                    del results['args']['vectors']

                with open(filename, 'w') as f:
                    json.dump(results, f, indent=4)
                
                print("--> Execution finished.")

            except Exception as e:
                print(f"Failure in {model_str} with {param_name}={val}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()