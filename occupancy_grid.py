import matplotlib.patches as patches
import numpy as np
import utils

from collections import defaultdict
from point_cloud import PointCloud2D
from shapely import geometry
from typing import List, Tuple
from occupancy_cell import OccupancyCell


class OccupancyGrid():

    def __init__(self, pcl_2d: PointCloud2D, resolution: float):
        pcl_2d_data = pcl_2d.data
        self.resolution = resolution
        self.x_min = np.min(pcl_2d_data[:, 0])
        self.x_max = np.max(pcl_2d_data[:, 0])
        self.cols = int(np.ceil((self.x_max - self.x_min) / self.resolution))

        self.y_min = np.min(pcl_2d_data[:, 1])
        self.y_max = np.max(pcl_2d_data[:, 1])
        self.rows = int(np.ceil((self.y_max - self.y_min) / self.resolution))

        self.x_min = np.min(pcl_2d_data[:, 0]) - (self.resolution / 2.0)
        self.x_max = np.max(pcl_2d_data[:, 0]) + (self.resolution / 2.0)
        self.y_min = np.min(pcl_2d_data[:, 1]) - (self.resolution / 2.0)
        self.y_max = np.max(pcl_2d_data[:, 1]) + (self.resolution / 2.0)

        self.grid = []
        self.populate_grid(pcl_2d_data)


    def populate_grid(self, pcl_2d_data: np.ndarray):
      for r in range(self.rows):
        self.grid.append([])
        for c in range(self.cols):
            x_lower = self.x_min + c * self.resolution
            x_upper = self.x_min + (c + 1) * self.resolution
            y_lower = self.y_min + r * self.resolution
            y_upper = self.y_min + (r + 1) * self.resolution

            x_points = np.logical_and(pcl_2d_data[:, 0] > x_lower, 
                                      pcl_2d_data[:, 0] <= x_upper)
            y_points = np.logical_and(pcl_2d_data[:, 1] > y_lower, 
                                      pcl_2d_data[:, 1] <= y_upper)

            cell_points = pcl_2d_data[np.logical_and(x_points, y_points), :].shape[0]

            x_center = x_lower + (self.resolution / 2.0)
            y_center = y_lower + (self.resolution / 2.0)
            is_wall = True if cell_points > 0 else False

            self.grid[r].append(OccupancyCell(x_center, y_center, self.resolution, r, c, is_wall))


    def get_cell(self, row: int, col: int) -> OccupancyCell:
      return self.grid[row][col]


    def set_occupied(self, cells:  List[OccupancyCell], occupied: bool = True):
      for cell in cells:
       cell.is_occupied = occupied


    def set_visited(self, cells:  List[OccupancyCell], visited: bool = True):
      for cell in cells:
       cell.is_visited = True


    def all_visited(self) -> bool:
      for r in range(self.rows):
        for c in range(self.cols):
          if not self.get_cell(r, c).is_occupied:
            return False

      return True


    def reset_visited(self, visited: bool = False) -> bool:
      for r in range(self.rows):
        for c in range(self.cols):
          self.get_cell(r, c).is_visited = visited


    def get_hamming_occupied_cells(self, cell_1: OccupancyCell, cell_2: OccupancyCell) -> List[OccupancyCell]:
      cells = []
      min_row = min(cell_1.row, cell_2.row)
      max_row = max(cell_1.row, cell_2.row)
      min_col = min(cell_1.col, cell_2.col)
      max_col = max(cell_1.col, cell_2.col)

      for r in range(min_row, max_row):
        for c in range(min_col, max_col):
          cell = self.get_cell(r, c)
          if cell.is_occupied and (cell != cell_1) and (cell != cell_2):
            cells.append(cell) 

      return cells


    def get_cell_from_coords(self, x: float, y: float) -> OccupancyCell:
      c = int(x - self.x_min / self.resolution)
      r = int(y - self.y_min / self.resolution)
      return self.get_cell(r, c)
    

    def is_occluded(self, cell_1: OccupancyCell, cell_2: OccupancyCell) -> bool:
      slopes = []


      cell_points = np.asarray(cell_2.poly.exterior.coords)

      for (x, y) in cell_points:
          slopes.append(utils.safe_slope(cell_1.x, cell_1.y, x, y))

      sorted_slope_idx = np.argsort(slopes)
      idx_1 = sorted_slope_idx[0]
      idx_2 = sorted_slope_idx[-1]
      seg_1 = geometry.LineString([cell_points[idx_1], (cell_1.x, cell_1.y)])
      seg_2 = geometry.LineString([cell_points[idx_2], (cell_1.x, cell_1.y)])

      for cell in self.get_hamming_occupied_cells(cell_1, cell_2):
        if seg_1.intersects(cell.poly) or seg_2.intersects(cell.poly):
          return True

      return False


    def plot(self, ax):
        for r in range(self.rows):
          for c in range(self.cols):
            self.get_cell(r, c).plot(ax)


    def in_grid_bounds(self, row, col):
      def in_bounds(v: float, v_min: float, v_max: float) -> bool:
        return ((v <= v_max) and (v >= v_min))

      return in_bounds(row, 0, self.rows - 1) and  in_bounds(col, 0, self.cols - 1)


    def get_neighbors(self, cell: OccupancyCell, 
                      include_diagonals: bool = False,  
                      filter_occupied: bool = False, 
                      filter_visited: bool = False,
                      filter_viewable: bool = False):

      horizontal_dirs = [
        [ 1,  0], # UP
        [ 0,  1], # RIGHT
        [-1,  0], # DOWN
        [ 0, -1]  # LEFT
      ]

      diagonal_dirs = [
        [ 1,  1], # UP RIGHT
        [-1,  1], # DOWN RIGHT
        [-1, -1], # DOWN LEFT
        [ 1, -1]  # UP LEFT
      ]

      dirs = horizontal_dirs + diagonal_dirs if include_diagonals else horizontal_dirs

      neighbors = []
      for d in dirs:
        r_new = cell.row + d[0]
        c_new = cell.col + d[1]

        if self.in_grid_bounds(r_new, c_new):
          neighbor = self.get_cell(r_new, c_new)

          if ((filter_occupied and neighbor.is_occupied) or 
              (filter_visited and neighbor.is_visited) or
              (filter_viewable and neighbor.is_viewable)):
            continue

          neighbors.append(neighbor)

      return neighbors


    def get_within_cells(self, poly: geometry.Polygon) -> List[OccupancyCell]:
      cells = []
      for r in range(self.rows):
        for c in range(self.cols):
          cell = self.get_cell(r, c)
          if (cell.poly.within(poly) or poly.within(cell.poly)):
            cells.append(cell)

      return cells


    def get_overlap_cells(self, poly: geometry.Polygon) -> List[OccupancyCell]:
      cells = []
      for r in range(self.rows):
        for c in range(self.cols):
          cell = self.get_cell(r, c)

          if (poly.overlaps(cell.poly) or cell.poly.within(poly)):
            cells.append(cell)

      return cells


    def mark_viewable_cells(self, start: OccupancyCell):

      for r in range(self.rows):
        for c in range(self.cols):
          self.get_cell(r, c).is_viewable = False

      queue = [start]
      viewed = defaultdict(bool)
      while len(queue):
        cell = queue.pop(0)
        cell.is_viewable = True

        for n in self.get_neighbors(cell, filter_viewable = True):

          if n.is_occupied:
            n.is_viewable = True

          elif not viewed[(n.row, n.col)]:
            queue.append(n)
            viewed[(n.row, n.col)] = True