from sklearn.linear_model import LinearRegression
import numpy as np
from typing import List

class MLService:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, features: List[float]) -> float:
        """Make prediction"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict([features])[0]
