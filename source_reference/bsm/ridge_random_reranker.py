"""Exact frozen class excerpt for inspection. Not executed in Phase1."""
import numpy as np

class RidgeRandomReranker:
    def __init__(self, seed: int, hidden: int = 32, ridge: float = 1.0) -> None:
        self.seed = seed
        self.hidden = hidden
        self.ridge = ridge
        self.mean = None
        self.std = None
        self.proj = None
        self.bias = None
        self.weight = None

    def _design(self, x: np.ndarray) -> np.ndarray:
        xs = (x - self.mean) / self.std
        h = np.maximum(0.0, xs @ self.proj + self.bias)
        return np.hstack([np.ones((xs.shape[0], 1)), xs, h])

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, float)
        y = np.asarray(y, float) * 2.0 - 1.0
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-6
        rng = np.random.default_rng(self.seed)
        self.proj = rng.normal(0.0, 1.0 / np.sqrt(max(x.shape[1], 1)), size=(x.shape[1], self.hidden))
        self.bias = rng.normal(0.0, 0.10, size=self.hidden)
        d = self._design(x)
        pos_w = max(1.0, float(np.sum(y <= 0)) / max(float(np.sum(y > 0)), 1.0))
        sw = np.where(y > 0, pos_w, 1.0)
        a = d * np.sqrt(sw)[:, None]
        b = y * np.sqrt(sw)
        penalty = np.eye(a.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        self.weight = np.linalg.pinv(a.T @ a + penalty) @ (a.T @ b)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._design(np.asarray(x, float)) @ self.weight

