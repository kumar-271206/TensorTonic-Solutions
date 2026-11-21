import numpy as np


def chi2_independence(C: list) -> dict:
  """Returns a dictionary with chi2 and expected."""
  observed = np.asarray(C, dtype=float)

  row_totals = observed.sum(axis=1)
  col_totals = observed.sum(axis=0)
  grand_total = observed.sum()

  expected = np.outer(row_totals, col_totals) / grand_total
  chi2 = float(np.sum(((observed - expected) ** 2) / expected))

  return {
      "chi2": chi2,
      "expected": expected,
  }