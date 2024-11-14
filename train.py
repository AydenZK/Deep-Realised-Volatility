import datetime as dt
import os
from pathlib import Path

import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler

from engineer import engineer_data
from models import LGBMModel, MLPModel
from utils import GridSearchResult, grid_search 

RESULTS_DIR = Path(os.getcwd()).joinpath("results")

MODELS = {
    "lgbm": LGBMModel,
    "mlp": MLPModel
}

HYPERPARAMS = {
    "lgbm": {
        "objective": ["rmse"], 
        "metric": ["rmse"], 
        'early_stopping_rounds': [75],
        "num_leaves": [500, 1000, 5000],
        "max_depth": [50, 100],
        "learning_rate": [0.001, 0.01, 0.1],
        "reg_alpha": [0, 0.01],
    },
    "mlp": {
        "hidden_layer_sizes": [
            (32, 64),
            (64, 128),
            (64, 64, 64),
            (64, 128, 128),
            (64, 128, 128, 32)
        ],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.00001, 0.0001, 0.001],
        "learning_rate": ["constant", "adaptive"],
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--model", type=str, default="lgbm", help="Model to train")
    parser.add_argument("--name", type=str, default=f"{dt.datetime.now():%Y%m%d_%H%M}_run", help="Label/Name for the training run")
    parser.add_argument("--grid_search", action="store_true", default=True, help="Perform Grid Search")
    parser.add_argument("--n_splits", type=int, default=2, help="Number of KFolds")
    parser.add_argument("--search_iter", type=int, default=2, help="Number of Grid Search Iterations")
    parser.add_argument("--load_local_features", action="store_true", default=False, help="Load Features previously saved locally")
    parser.add_argument("--normalize", action="store_true", default=False, help="Normalize Data")
    parser.add_argument("--drop_na", action="store_true", default=False, help="Drop NA Values")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of Jobs")
    args = parser.parse_args()
    return args

def train():
    args = parse_args()

    if args.model not in MODELS or args.model not in HYPERPARAMS:
        raise ValueError(f"Invalid Model: {args.model}")
    model_class = MODELS[args.model]
    params = HYPERPARAMS[args.model]

    df_train, _ = engineer_data(load_local=args.load_local_features)
    
    X = df_train.drop(columns=["row_id", "time_id", "target"]).copy()
    y = df_train["target"].copy()

    if args.drop_na or args.model == "mlp":
        na_idx = X.isna().any(axis=1)
        X = X[~na_idx].reset_index(drop=True)
        y = y[~na_idx].reset_index(drop=True)
    if args.normalize or args.model == "mlp":
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X))

    n_splits = args.n_splits if args.grid_search else 2
    grid_search_iter = args.search_iter if args.grid_search else 2

    grid_search_results: GridSearchResult = grid_search(
        _model_class = model_class,
        X = X,
        y = y,
        search_grid=params,
        cv=n_splits,
        n_jobs=args.n_jobs,
        n_iter=grid_search_iter
    )

    grid_search_results.save(name=RESULTS_DIR.joinpath(f"{args.name}"))

if __name__ == "__main__":
    train()