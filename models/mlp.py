from sklearn.neural_network import MLPRegressor

from .base_model import BaseModel


class MLPModel(BaseModel):
    def __init__(self, params: dict):
        self.params = params
        self.model = None
        self.trained = False

    def train(self, X_train, y_train, X_valid, y_valid):
        self.model = MLPRegressor(**self.params)
        self.model.fit(X_train, y_train)
        self.trained = True