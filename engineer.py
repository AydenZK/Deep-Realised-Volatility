import os
import itertools
from pathlib import Path
from typing import List, Tuple
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
import pandas as pd

DATA_DIR = Path(os.getcwd()).joinpath("data")

RVOL_FEATURE_GRID = {
    "wap2_wt": [0, 0.15],
    "ewa_alpha": [0.2, 0.6, 1],
    "shift_size": [1, 3, 5] 
}

def wap_ewa(df: pd.DataFrame, wap2_wt: float, ewa_alpha: float) -> pd.DataFrame:
    if "wap1" not in df.columns and "wap2" not in df.columns:
        df['wap1'] = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
        df['wap2'] = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    df[f'wap_{wap2_wt}_{ewa_alpha}'] = (df['wap1'] * (1 - wap2_wt) + df['wap2'] * wap2_wt).ewm(alpha=ewa_alpha).mean()
    return df

def spread_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df['spread_ratio'] = ((df['ask_price1'] / df['bid_price1'] - 1) * 10000)
    return df

def volume_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
    return df

def calc_rvol(x: pd.Series, shift_size: int) -> float:
    x = np.log(x).diff(shift_size)
    return np.sqrt(np.sum(x**2))

def create_rvol_calc(shift_size: int) -> callable:
    return lambda x: calc_rvol(x, shift_size)

def engineer_book_features(stock_id: int, train_test: str) -> pd.DataFrame:
    file_path = load_source_data(stock_id, "book", train_test)
    df_book = pd.read_parquet(file_path)

    for wap2_wt, ewa_alpha in itertools.product(*list(RVOL_FEATURE_GRID.values())[:2]):
        df_book = wap_ewa(df_book, wap2_wt, ewa_alpha)

    df_book = spread_ratio(df_book)
    df_book = volume_imbalance(df_book)
    
    agg_dict = {
        'spread_ratio_mean': ('spread_ratio', 'mean'),
        'spread_ratio_std': ('spread_ratio', 'std'),
        'volume_imbalance': ('volume_imbalance', 'mean')
    }

    for wap2_wt, ewa_alpha, shift_size in itertools.product(*RVOL_FEATURE_GRID.values()):
        agg_dict[f'rvol_{wap2_wt}_{ewa_alpha}_{shift_size}'] = (f"wap_{wap2_wt}_{ewa_alpha}", create_rvol_calc(shift_size))

    df_book_features = df_book.groupby('time_id').agg(**agg_dict).reset_index()
    df_book_features['stock_id'] = stock_id
    
    return df_book_features

def engineer_trade_features(stock_id: int, train_test: str) -> pd.DataFrame:
    file_path = load_source_data(stock_id, "trade", train_test)
    df_trade = pd.read_parquet(file_path)

    df_trade_features = df_trade.groupby('time_id').agg(
        trade_volume = ('size', 'sum'),
        trade_count = ('order_count', 'sum'),
    ).reset_index()
    df_trade_features['stock_id'] = stock_id

    return df_trade_features

def load_source_data(stock_id: int, data_type: str, train_test: str) -> Path:
    file_dir = DATA_DIR.joinpath(f"{data_type}_{train_test}.parquet").joinpath(f"stock_id={stock_id}")
    file_name = os.listdir(file_dir)[0]
    full_path = file_dir.joinpath(file_name)
    return full_path

def for_joblib(stock_id: int, train_test: str) -> pd.DataFrame:
    df_bk = engineer_book_features(stock_id, train_test)
    df_tr = engineer_trade_features(stock_id, train_test)
    
    df_tmp = pd.merge(df_bk, df_tr, on=['stock_id', 'time_id'], how='left')

    return df_tmp

def engineer_features(stock_ids: List[int], train_test: str) -> pd.DataFrame:
    dfs = Parallel(n_jobs=-1, verbose=1)(delayed(for_joblib)(stock_id, train_test) for stock_id in tqdm(stock_ids))
    df = pd.concat(dfs, ignore_index=True)
    df["row_id"] = df["stock_id"].astype(str) + "-" + df["time_id"].astype(str)
    return df

def engineer_data(load_local: bool = False, save: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if load_local:
        df_train = pd.read_parquet(DATA_DIR.joinpath("train_features.parquet"))
        df_test = pd.read_parquet(DATA_DIR.joinpath("test_features.parquet"))
        return df_train, df_test
    
    df_train_targets = pd.read_csv(DATA_DIR.joinpath("train.csv"))
    train_ids = df_train_targets["stock_id"].unique()
    df_train = engineer_features(train_ids, "train")
    
    df_train = pd.merge(df_train, df_train_targets, on=['stock_id', 'time_id'], how='inner')

    test_df = pd.read_csv(DATA_DIR.joinpath("test.csv"))
    test_ids = test_df["stock_id"].unique()
    df_test = engineer_features(test_ids, "test")

    if save:
        df_train.to_parquet(DATA_DIR.joinpath("train_features.parquet"))
        df_test.to_parquet(DATA_DIR.joinpath("test_features.parquet"))

    return df_train, df_test

# if __name__ == "__main__":
#     # Save data for ease of use in the future
#     df_train, df_test = engineer_data(load_local=False, save=True)
#     print(df_train.head(), len(df_train))
#     print(df_test.head(), len(df_test))