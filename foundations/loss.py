import numpy as np
from numpy.typing import NDArray

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7
        y_pred_safe = np.clip(y_pred, epsilon, 1 - epsilon)

        losses = y_true * np.log(y_pred_safe) + (1 - y_true) * np.log(1 - y_pred_safe)
        ans = -np.mean(losses)
        
        return round(float(ans), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7
        safe = np.clip(y_pred, epsilon, 1.0)
        sample_losses = np.sum(y_true * np.log(safe), axis=1)
        print(sample_losses)
        ans = -np.mean(sample_losses)
        
        return round(float(ans), 4)

