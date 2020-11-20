import numpy as np


def safe_slope(x1: float, y1: float, x2: float, y2: float) -> float:
    x_diff = (x2 - x1)
    if x_diff == 0:
      return np.inf

    return (y2 - y1) / x_diff 