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

        # The number of rows and columns are determined by the resolution. Similarly
        # the boundaries of the grid are buffered with the resolution to ensure that
        # all points are contained and accounted for.
        self.x_min = np.min(pcl_2d_data[:, 0]) - self.resolution
        self.x_max = np.max(pcl_2d_data[:, 0]) + self.resolution
        self.cols = int(np.ceil((self.x_max - self.x_min) / self.resolution))

        self.y_min = np.min(pcl_2d_data[:, 1]) - self.resolution
        self.y_max = np.max(pcl_2d_data[:, 1]) + self.resolution
        self.rows = int(np.ceil((self.y_max - self.y_min) / self.resolution))

        self.grid = []
        self.populate_grid(pcl_2d_data)


    def populate_grid(self, pcl_2d_data: np.ndarray):
      """Accepts a list of 2D points and discretizes them into a 2D grid using the provided resolutions
      set in the class constructor."""
      for r in range(self.rows):
        self.grid.append([])
        for c in range(self.cols):

            # Each grid cell is a 2D rectangle.
            x_lower = self.x_min + c * self.resolution
            x_upper = self.x_min + (c + 1) * self.resolution
            y_lower = self.y_min + r * self.resolution
            y_upper = self.y_min + (r + 1) * self.resolution


            # If there exist point cloud points within the boundaries of the cell then it is marked
            # as occupied.
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
      """Retrieves a reference to a grid cell."""
      if self.in_grid_bounds(row, col):
        return self.grid[row][col]

      return None

    def set_occupied(self, cells:  List[OccupancyCell], occupied: bool = True):
      """Sets the occupancy state of a list of grid cells."""
      for cell in cells:
       cell.is_occupied = occupied


    def set_visited(self, cells:  List[OccupancyCell], visited: bool = True):
      """Sets the is visited state of a list of grid cells."""
      for cell in cells:
       cell.is_visited = True


    def get_hamming_occupied_cells(self, cell_1: OccupancyCell, cell_2: OccupancyCell) -> List[OccupancyCell]:
      """Gets all cells within a rectangular bounded region between cell_1 and cell_2 but also exlcuding them."""
      cells = []
      min_row = min(cell_1.row, cell_2.row)
      max_row = max(cell_1.row, cell_2.row)
      min_col = min(cell_1.col, cell_2.col)
      max_col = max(cell_1.col, cell_2.col)

      for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
          cell = self.get_cell(r, c)

          if cell.is_occupied and (cell != cell_1) and (cell != cell_2):
            cells.append(cell) 

      return cells


    def get_cell_from_coords(self, x: float, y: float) -> OccupancyCell:
      """Gets the cell from the grid containing the corresponding x, y point."""
      c = int((x - self.x_min) / self.resolution)
      r = int((y - self.y_min) / self.resolution)
      return self.get_cell(r, c)
    

    def is_occluded(self, cell_1: OccupancyCell, cell_2: OccupancyCell) -> bool:
      """Determines whether cell_2 is occluded when observing it from cell_1."""

      neighbors = self.get_neighbors(cell_2)
      point_slopes = [] 
      distances = []

      for (x1, y1), (x2, y2), (row_offset, col_offset) in cell_2.get_sides():
        neighbor = self.get_cell(cell_2.row + row_offset, cell_2.col + col_offset)
        if neighbor is None:
          continue

        # Determines the distance from the centroid of an edge of cell_2 to the center of cell_1.
        side_centroid_x = (x1 + x2) / 2.0
        side_centroid_y = (y1 + y2) / 2.0
        side_distance = utils.euclidean_distance(side_centroid_x, side_centroid_y, cell_1.x, cell_1.y)
        distances.append(side_distance)

        # Calculates the slopes between the corners of the side and the center of cell_1.
        s1 = utils.safe_slope(cell_1.x, cell_1.y, x1, y1)
        s2 = utils.safe_slope(cell_1.x, cell_1.y, x2, y2)
        point_slopes.append([s1, s2, (x1, y1), (x2, y2), not neighbor.is_occupied and neighbor.is_viewable])


      # Identifies the two lines between corners connected to oncuppied edges of cell_2 and the center of cell_1
      # that maximize the area of the polygon formed by connecting the points. This is to ensure that we check
      # for full occlusion.
      sorted_distances_idx = np.argsort(distances)
      idx_1 = sorted_distances_idx[0]
      idx_2 = sorted_distances_idx[-1]
      free_slopes = []
      free_points = []


      # Determines which edges have oncuppied neighbors to identify the area required for viewing.
      if point_slopes[idx_1][4]:
        free_slopes.append(point_slopes[idx_1][0])
        free_slopes.append(point_slopes[idx_1][1])
        free_points.append(point_slopes[idx_1][2])
        free_points.append(point_slopes[idx_1][3])

      if point_slopes[idx_2][4]:
        free_slopes.append(point_slopes[idx_2][0])
        free_slopes.append(point_slopes[idx_2][1])
        free_points.append(point_slopes[idx_2][2])
        free_points.append(point_slopes[idx_2][3])

      if not len(free_slopes):
        return False

      # Now that the continous and oncuppied edges are identified we redraw the largest possible polygon
      # between the identified cell_2 corners and the center of cell_1.
      sorted_slope_idx = np.argsort(free_slopes)
      idx_1 = sorted_slope_idx[0]
      idx_2 = sorted_slope_idx[-1]

      
      x1, y1 = free_points[idx_1]
      x2, y2 = free_points[idx_2]

      seg_1 = geometry.LineString([(x1, y1), (cell_1.x, cell_1.y)])
      seg_2 = geometry.LineString([(x2, y2), (cell_1.x, cell_1.y)])


      # Any occupied cells that intersect the lines / polygon drawn between the corners of cell_2 and the center
      # of cell_1 represent that cell_2 is occluded.
      for cell in self.get_hamming_occupied_cells(cell_1, cell_2):
        if seg_1.intersects(cell.poly) or seg_2.intersects(cell.poly):
          return True

      return False


    def plot(self, ax):
      """Plots all cells within the grid on a provided set of axeses"""
      for r in range(self.rows):
        for c in range(self.cols):
          self.get_cell(r, c).plot(ax)

      ax.set_xlim(self.x_min, self.x_max)
      ax.set_ylim(self.y_min, self.y_max)


    def in_grid_bounds(self, row, col):
      """Determines whether a given row and col location is within the boundaries
      of the grid."""

      def in_bounds(v: float, v_min: float, v_max: float) -> bool:
        return ((v <= v_max) and (v >= v_min))

      return in_bounds(row, 0, self.rows - 1) and  in_bounds(col, 0, self.cols - 1)


    def get_neighbors(self, cell: OccupancyCell, 
                      include_diagonals: bool = False,  
                      filter_occupied: bool = False, 
                      filter_visited: bool = False,
                      filter_viewable: bool = False):
      """Gets all neighbors of the provided cell. Horizontal neighbors are included by default
      but diagonal neighbors can be added. Additionally, numerous filters can be applied to
      the neighbors to determine the relevant set of nearby cells."""
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

      # Maybe account for diagonal neighbors.
      dirs = horizontal_dirs + diagonal_dirs if include_diagonals else horizontal_dirs

      neighbors = []
      for d in dirs:
        r_new = cell.row + d[0]
        c_new = cell.col + d[1]

        if self.in_grid_bounds(r_new, c_new):
          neighbor = self.get_cell(r_new, c_new)

          # Apply neighbor filters.
          if ((filter_occupied and neighbor.is_occupied) or 
              (filter_visited and neighbor.is_visited) or
              (filter_viewable and neighbor.is_viewable)):
            continue

          neighbors.append(neighbor)

      return neighbors


    def get_within_cells(self, poly: geometry.Polygon) -> List[OccupancyCell]:
      """Determines which cells are contained by or contain the provided polygon."""
      cells = []
      for r in range(self.rows):
        for c in range(self.cols):
          cell = self.get_cell(r, c)

          if (cell.poly.within(poly) or poly.within(cell.poly)):
            cells.append(cell)

      return cells


    def get_overlap_cells(self, poly: geometry.Polygon) -> List[OccupancyCell]:
      """Determines which cells in the grid intersect the provided polygon. This will return no
      cells if the polygon is smaller then a single cell and contained within it."""

      cells = []
      for r in range(self.rows):
        for c in range(self.cols):
          cell = self.get_cell(r, c)
          if (poly.overlaps(cell.poly) or cell.poly.within(poly)):
            cells.append(cell)

      return cells


    def print_coverage(self):
      """Prints the number of viewed or visited cells within the map at a given
      point in time versus the total number of viewable cells."""
      total = 0
      viewed = 0
      for r in range(self.rows):
        for c in range(self.cols):
          cell = self.get_cell(r, c)
          if cell.is_viewable:
            total += 1
          if cell.is_visited:
            viewed += 1
      print("Coverage %d / %d" % (viewed, total))


    def mark_viewable_cells(self, start: OccupancyCell):
      """Traverses the grid using BFS starting at the provided start cell and determines which cells
      are physically accessible or connected and not blocked by occupied cells. This does not take into
      account the size of the robot or object trying to traverse the pass, but merely determines if a path
      exists however big or small. This is used to initialize the map to determine when the goal state of full
      coverage is complete."""
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