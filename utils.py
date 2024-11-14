import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from tqdm import tqdm

from models import BaseModel

RESULTS_DIR = Path(os.getcwd()).joinpath("results")

@dataclass
class ModelResult:
    model: BaseModel
    train_rmspe: float
    valid_rmspe: float


@dataclass
class GridSearchResult:
    search_results: pd.DataFrame
    best_model: ModelResult

    def save(self, name: str):
        self.search_results.to_csv(RESULTS_DIR.joinpath("grid_searches").joinpath(f"{name}.csv"))
        self.best_model.model.save(path=RESULTS_DIR.joinpath("models").joinpath(name))


def rmspe(y_true, y_pred):
    return (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))

def cv_train(
    _model_class: Type[BaseModel],
    kf: KFold, 
    X: pd.DataFrame, 
    y: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[Dict[str, Any], ModelResult]:
    """Cross Validation Run"""
    cv_models: List[ModelResult] = []
    model = _model_class(params)

    for train_idx, valid_idx in kf.split(X):
        X_train, y_train = X.loc[train_idx], y[train_idx]
        X_valid, y_valid = X.loc[valid_idx], y[valid_idx]

        model.train(X_train, y_train, X_valid, y_valid)
        model_res = ModelResult(
            model=model,
            train_rmspe=rmspe(y_train, model.predict(X_train)),
            valid_rmspe=rmspe(y_valid, model.predict(X_valid))
        )
        cv_models.append(model_res)

    cv_results = params.copy()
    cv_results.update({
        "train_rmspe_mean": np.mean([m.train_rmspe for m in cv_models]),
        "valid_rmspe_mean": np.mean([m.valid_rmspe for m in cv_models]),
        "train_rmspe_std": np.std([m.train_rmspe for m in cv_models]),
        "valid_rmspe_std": np.std([m.valid_rmspe for m in cv_models])
    })
    best_cv_model = cv_models[np.argmax([m.valid_rmspe for m in cv_models])]
    
    return GridSearchResult(pd.DataFrame([cv_results]), best_cv_model)


def grid_search(
    _model_class: Type[BaseModel],
    X: pd.DataFrame, 
    y: pd.DataFrame,
    search_grid: Dict[str, List[Any]],
    n_iter: int, 
    cv: int = 2, 
    randomise: bool = True,
    n_jobs: int = -1
) -> GridSearchResult:
    """ Generalized Grid Searcher"""
    
    # Construct the search grid
    param_combinations = [
        dict(zip(search_grid.keys(), p))
        for p in itertools.product(*search_grid.values())
    ]
    if randomise:
        param_combinations = np.random.choice(param_combinations, n_iter)

    # Setup the cross validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Run the grid search
    results: List[GridSearchResult] = Parallel(n_jobs=-n_jobs)(delayed(cv_train)(_model_class, kf, X, y, params) for params in tqdm(param_combinations))

    # Collect the results
    search_results = pd.concat([r.search_results for r in results], ignore_index=True)
    best_model = results[np.argmax([r.best_model.valid_rmspe for r in results])].best_model

    return GridSearchResult(search_results, best_model)

def calc_model_importance(model, feature_names=None, importance_type='gain'):
    importance_df = pd.DataFrame(model.feature_importance(importance_type=importance_type),
                                 index=feature_names,
                                 columns=['importance']).sort_values('importance')
    return importance_df

def plot_importance(importance_df, title='',
                    save_filepath=None, figsize=(8, 12)):
    _, ax = plt.subplots(figsize=figsize)
    importance_df.plot.barh(ax=ax)
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_filepath is None:
        plt.show()
    else:
        plt.savefig(save_filepath)
    plt.close()