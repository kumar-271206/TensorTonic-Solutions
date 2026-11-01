import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    arr = np.array(matrix, dtype=float)
    
    if norm_type == "l1":
        norm = np.sum(np.abs(arr), axis=axis, keepdims=True)
    elif norm_type == "l2":
        norm = np.sqrt(np.sum(arr ** 2, axis=axis, keepdims=True))
    elif norm_type == "max":
        norm = np.max(np.abs(arr), axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")
        
    norm = np.where(norm == 0, 1.0, norm)
    return arr / norm