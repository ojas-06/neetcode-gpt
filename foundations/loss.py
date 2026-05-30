import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred_safe = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # 2. Vectorized element-wise math (no loops!)
        losses = y_true * np.log(y_pred_safe) + (1 - y_true) * np.log(1 - y_pred_safe)
        
        # 3. np.mean() automatically sums the array and divides by N
        ans = -np.mean(losses)
        
        return round(float(ans), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        safe = np.clip(y_pred, epsilon, 1.0)
        sample_losses = np.sum(y_true * np.log(safe), axis=1)
    
    # 3. Average the losses across all samples
        ans = -np.mean(sample_losses)
        
        return round(float(ans), 4)

