import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """Returns a NumPy array with shape (n_test, k)."""
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    if X_train.ndim == 1:
        X_train = X_train[:, None]
    if X_test.ndim == 1:
        X_test = X_test[:, None]
        
    n_train = X_train.shape[0]
    effective_k = min(k, n_train)
    
    diff = X_test[:, None, :] - X_train[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    
    sorted_indices = np.argsort(distances, axis=1)
    
    if effective_k < k:
        top_k = sorted_indices[:, :effective_k]
        padding = np.full((X_test.shape[0], k - effective_k), -1, dtype=int)
        return np.hstack([top_k, padding])
    
    return sorted_indices[:, :k]