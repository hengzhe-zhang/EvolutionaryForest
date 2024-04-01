from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


class DecisionTool:
    def __init__(self, top_k: int = 1):
        self.data_queue: List[Tuple[np.ndarray, np.ndarray]] = []
        self.learner = None
        self.top_k = top_k

    def append(self, semantics: np.ndarray, data: np.ndarray):
        top_k_index = self.top_k_indices(data)
        self.data_queue.append((semantics, top_k_index))

    def top_k_indices(self, data: np.ndarray) -> np.ndarray:
        return np.argsort(data)[-self.top_k :]

    def fit(self):
        self.learner = MultiOutputClassifier(LogisticRegression(), n_jobs=-1)
        # Assuming each 'semantics' is a feature vector and corresponding 'top_k_index' are the labels
        X = np.array([item[0] for item in self.data_queue])
        Y = np.array([item[1] for item in self.data_queue])
        self.learner.fit(X, Y)

    def predict(self, semantics: np.ndarray, index=0) -> np.ndarray:
        # Assuming 'semantics' needs to be reshaped as a single sample for prediction
        semantics_reshaped = semantics.reshape(1, -1)
        probs = self.learner.predict_proba(semantics_reshaped)
        # Return indices of top predictions
        return self.top_k_indices_from_probs(probs[index])

    def top_k_indices_from_probs(self, probs: np.ndarray) -> np.ndarray:
        # Assuming 'probs' is an array of class probabilities; this method needs adjustment based on 'learner' output
        # Here we assume a simple mechanism for demonstration purposes
        top_k_probs = np.argsort(probs)[-self.top_k :]
        return top_k_probs

    def decision(self, semantics: np.ndarray, index) -> bool:
        return index in self.predict(semantics)
