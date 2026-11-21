import numpy as np


def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
  """Returns a causally masked NumPy array matching the shape of scores."""
  arr = np.asarray(scores, dtype=np.float64)
  seq_len = arr.shape[-1]

  # True only for entries strictly above the main diagonal (j > i)
  future_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

  # np.where broadcasts future_mask across any leading batch or head dimensions
  # and returns a new array without mutating `scores` or `arr`.
  return np.where(future_mask, float(mask_value), arr)