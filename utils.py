import math
import numpy as np


def safe_slope(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calulates the slope between two points in a way tha avoids division by zeros."""
    x_diff = (x2 - x1)
    
    if x_diff == 0:
        # Avoid division by zero by return inf.
        return np.inf

    return (y2 - y1) / x_diff 


def z_axis_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calulates that angle in degrees [0, 360] that the slope of the provided points
    form with rotation about the z axis."""

    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))  

    # Wrap angle to bound it between 0 and 360.
    if angle < 0:
        angle += 360

    if angle > 360:
        angle -= 360

    return angle


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculates the euclidean distance between two points."""
    return math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))


def wrap_heading(current_heading: float, new_heading: float) -> float:
    """Determines the offset between the current and provided heading to adjust
    them such that the normal subtraction between them results in an angle less than
    or equal to 180 degrees."""
    if abs(new_heading - current_heading) > 180:
      if new_heading > current_heading:
        new_heading -= 360

      else:
        new_heading += 360

    return new_heading


def angular_distance(a1: float, a2: float) -> float:
    """Calculates the angular distance between two angles in degrees."""
    phi = abs(a2 - a1) % 360;      
    distance = 360 - phi if phi > 180 else phi;
    return distance;
