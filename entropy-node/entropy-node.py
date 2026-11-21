import numpy as np


def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    if len(y) == 0:
        return 0.0

    arr = np.asarray(y)
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / arr.size

    # Only compute log2 on positive probabilities to avoid log2(0)
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)