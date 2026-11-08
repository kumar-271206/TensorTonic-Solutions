import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X, where missing values 
    are imputed with the column-wise mean or median.
    """
    # Convert input to a floating-point NumPy array to support NaNs
    arr = np.array(X, dtype=float)
    
    # Handle 1D input by reshaping to 2D column vector for uniform processing
    is_1d = (arr.ndim == 1)
    if is_1d:
        arr = arr[:, np.newaxis]
        
    out = arr.copy()
    
    # Iterate over each feature/column
    for j in range(arr.shape[1]):
        col = arr[:, j]
        mask_nan = np.isnan(col)
        valid_vals = col[~mask_nan]
        
        if valid_vals.size == 0:
            # If the feature contains only missing values, fill with 0.0
            fill_val = 0.0
        else:
            if strategy == "median":
                fill_val = np.median(valid_vals)
            else:
                fill_val = np.mean(valid_vals)
                
        # Replace NaN values with the computed statistic
        out[mask_nan, j] = fill_val
        
    # Return 1D if input was originally 1D, otherwise return 2D
    return out.squeeze() if is_1d else out