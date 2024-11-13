from abc import ABC, abstractmethod
import numpy as np
import pickle

class BaseModel(ABC):
    """ Base class for all prediction moodels
    """
    model = None
    trained = False

    @abstractmethod
    def train(self, X_train, y_train, X_valid, y_valid):
        pass
    
    @staticmethod
    def load(path: str) -> 'BaseModel':
        with open(f"{path}.pkl", "rb") as f:
            return pickle.load(f)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str):
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(self, f)

