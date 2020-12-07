import math
import matplotlib.patches as patches
import numpy as np
import utils
from shapely import geometry
from typing import List, Tuple
from collections import defaultdict


class OccupancyCell():

  def __init__(self, x: float, y: float, w: float, row: int, col: int, is_occupied: bool):
    self.x = x 
    self.y = y
    self.w = w
    self.row = row
    self.col = col
    self.x_lower = self.x - (self.w / 2.0)
    self.x_upper = self.x + (self.w / 2.0)
    self.y_lower = self.y - (self.w / 2.0)
    self.y_upper = self.y + (self.w / 2.0)

    self.poly = geometry.Polygon([(self.x_upper, self.y_upper),  
                                  (self.x_upper, self.y_lower),
                                  (self.x_lower, self.y_lower), 
                                  (self.x_lower, self.y_upper)])

    self.is_occupied = is_occupied
    self.is_visited = False
    self.is_frontier = False
    self.is_viewable = True
    self.is_frontier = False


  def key(self) -> Tuple[int, int]:
    """Determines a unique key for the provided occupancy cell."""
    return (self.row, self.col)


  def plot(self, ax):
    """Plots the occupancy cell on a set of axeses and uses colors and edge sides
    to reflect the state of the cell."""
    plot_params = {
      'linewidth': 0.1,
      'fill': False,
      'alpha': 0.8,
      'edgecolor': 'k'
    }

    if self.is_occupied:
      plot_params['linewidth'] = 3
      plot_params['edgecolor'] = 'k'

    if self.is_visited:
      plot_params['facecolor'] = 'g'
      plot_params['fill'] = True

    if not self.is_viewable:
      plot_params['facecolor'] = 'b'
      plot_params['fill'] = True

    if self.is_frontier:
      plot_params['facecolor'] = 'y'
      plot_params['fill'] = True

    rect = patches.Rectangle((self.x_lower, self.y_lower),
                             self.w,
                             self.w,
                             **plot_params)
    ax.add_patch(rect)


  def get_sides(self, eps: float = 0.005) -> List[Tuple[float]]:
    """Gets the two corners corresponding to each side of the occupancy cell. An eps parameter is provided to
    move the corners on a given side closer to each other by eps along the shared axis. This is used to prevent
    intersections with bordering polygons when performing occlusion calculations."""
    return [[(self.x_upper, self.y_upper - eps), (self.x_upper, self.y_lower + eps), [0,  1]], # RIGHT
            [(self.x_upper - eps, self.y_lower), (self.x_lower + eps, self.y_lower), [-1, 0]], # DOWN
            [(self.x_lower, self.y_lower + eps), (self.x_lower, self.y_upper - eps), [0, -1]], # LEFT
            [(self.x_lower + eps, self.y_upper), (self.x_upper - eps, self.y_upper), [1,  0]]] # UP


  def print(self, prefix: str = "Cell"):
    """Prints the state of the occupancy cell to standard output."""
    print("%s | Row: %d | Col %d | X: %0.1f | Y: %0.1f | Visit: %d | Occupy: %d | View: %d | Frontier: %d |" % 
          (prefix, self.row, self.col, self.x, self.y, self.is_visited, self.is_occupied, self.is_viewable, self.is_frontier))


  def __eq__(self, cell) -> bool:
    """Overrides the equality operator for the occupancy cell which confirms whether their unique keys are equal."""
    return self.key() == cell.key()


  def euclidean_distance(self, cell) -> float:
    """Determines the euclidean distance between two occupancy cells."""
    return utils.euclidean_distance(cell.x, cell.y, self.x, self.y)


  def z_axis_angle(self, cell) -> float:
    """Gets the angle between the current cell and the provided cell by determining the rotation of the connecting line
    about the z-axis in a counter-clockwise fashion."""
    return utils.z_axis_angle(self.x, self.y, cell.x, cell.y)
