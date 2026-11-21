import numpy as np


def matrix_transpose(A: list) -> np.ndarray:
  """Returns the transposed matrix as a NumPy array."""
  if not A or not A[0]:
    return np.empty((0, 0))

  n = len(A)
  m = len(A[0])
  result = np.empty((m, n), dtype=np.asarray(A).dtype)

  for i in range(n):
    for j in range(m):
      result[j, i] = A[i][j]

  return result