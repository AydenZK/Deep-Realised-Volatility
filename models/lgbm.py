
import lightgbm as lgb
import numpy as np

from .base_model import BaseModel

def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


class LGBMModel(BaseModel):
    def __init__(self, params: dict):
        self.params = params
        self.model = None
        self.trained = False

    @staticmethod
    def feval_rmspe(y_pred, lgb_train: lgb.Dataset):
        y_true = lgb_train.get_label()
        return 'RMSPE', rmspe(y_true, y_pred), False

    def train(self, X_train, y_train, X_valid, y_valid):
        X_train_lgb = lgb.Dataset(
            X_train, y_train, 
            categorical_feature=["stock_id"], 
            weight=1/np.square(y_train)
        )
        X_valid_lgb = lgb.Dataset(
            X_valid, y_valid, 
            categorical_feature=["stock_id"], 
            weight=1/np.square(y_valid)
        )

        self.model = lgb.train(
            self.params, 
            X_train_lgb, 
            valid_sets=[X_train_lgb, X_valid_lgb], 
            num_boost_round=1000, 
            feval=self.feval_rmspe
        )
        self.trained = True
    

    