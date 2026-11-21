import numpy as np


def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    mean_x = float(np.mean(arr))

    # Bessel's correction (ddof=1) for sample standard deviation
    s = float(np.std(arr, ddof=1))

    if s == 0.0:
        if mean_x == mu0:
            return 0.0
        return float(np.inf) if mean_x > mu0 else float(-np.inf)

    se = s / np.sqrt(n)
    t_stat = (mean_x - mu0) / se
    return float(t_stat)