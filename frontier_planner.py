import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import utils
from occupancy_grid import OccupancyCell, OccupancyGrid
from robot import Robot
from shapely import affinity, geometry
from typing import List, Tuple


class FrontierPlanner():

  def __init__(self, grid: OccupancyGrid, robot: Robot):
    self.grid = grid
    self.robot = robot
    self.frontier = {}
    self.coverage_path = []


  def get_shortest_path_bfs(self, start: OccupancyCell, goal: OccupancyCell) -> Tuple[List[OccupancyCell], float]:
    distances = {}
    path_history = {}

    distance = 0
    path = []
    angles = []

    current = None
    if self.is_visible(start, goal):
      camera_poly = copy.deepcopy(self.robot.camera_poly) 
      x_offset = goal.x - start.x
      y_offset = goal.y - start.y
      camera_poly = affinity.translate(camera_poly, xoff = x_offset, yoff=y_offset)
      slope = utils.safe_slope(start.x, start.y, goal.x, goal.y)
      angle = math.degrees(math.atan(slope))

      path = [start]
      angles = [angle]
      distance = 0
      return path, angles, distance



    queue = [start]

    while len(queue):
      cell = queue.pop(0)


      for n in self.grid.get_neighbors(cell, include_diagonals=True): 
        if n.key() in path_history:
          continue

        path_history[n.key()] = cell

        if (not self.is_collision(n)):
          d = cell.euclidean_distance(n)
          distances[n.key()] = d
          if self.is_visible(n, goal):
            current = n
            break

          queue.append(n)



    camera_poly = copy.deepcopy(self.robot.camera_poly) 
    x_offset = goal.x - current.x
    y_offset = goal.y - current.y
    camera_poly = affinity.translate(camera_poly, xoff = x_offset, yoff=y_offset)
    slope = utils.safe_slope(current.x, current.y, goal.x, goal.y)
    angle = math.degrees(math.atan(slope))

    path = [current]
    angles = [angle]

    while current.key() in path_history:
      path.insert(0, cell)
      next_cell = path_history[cell.key()]
      camera_poly = copy.deepcopy(self.robot.camera_poly) 
      x_offset = current.x - next_cell.x
      y_offset = current.y - next_cell.y
      camera_poly = affinity.translate(camera_poly, xoff = x_offset, yoff=y_offset)
      
      slope = utils.safe_slope(next_cell.x, next_cell.y, current.x, current.y)
      angle = math.degrees(math.atan(slope))
      angles.insert(0, angle)

      distance += distances[current.key()]
      cell = path_history[current.key()]

    return path, angles, distance


  def is_visible(self, current: OccupancyCell, goal: OccupancyCell) -> bool:

    camera_poly = copy.deepcopy(self.robot.camera_poly) 
    x_offset = goal.x - current.x
    y_offset = goal.y - current.y
    camera_poly = affinity.translate(camera_poly, xoff = x_offset, yoff=y_offset)


    slope = utils.safe_slope(current.x, current.y, goal.x, goal.y)
    angle = math.degrees(math.atan(slope))
    camera_poly = affinity.rotate(camera_poly, angle, (current.x, current.y))
    
    if camera_poly.overlaps(goal.poly) and not self.grid.is_occluded(current, goal):
      return True

    return False


    def get_viewing_angle(self, x: float, y) -> float:
        return self.get_angle(x, y) - self.yaw

    def in_view(self, view_angle: float) -> bool:

        if abs(view_angle) < (self.sensor_view_angle / 2.0):
            return True

        return False


  def is_collision(self, cell: OccupancyCell) -> bool:
    new_body_poly = geometry.Point(cell.x, cell.y).buffer(self.robot.radius)
    cells = self.grid.get_overlap_cells(new_body_poly)
    for cell in cells:
      if (cell.is_visited and cell.is_occupied):
        return True

    return False


  def maybe_visit_and_update_frontier(self, cells: List[OccupancyCell]):
    for cell in cells:
      if cell.is_visited:
        return

      cell.is_visited = True
      cell.is_frontier = False
      self.frontier.pop(cell.key(), None)

      for n in self.grid.get_neighbors(cell):
        n_key = n.key()
        if not n.is_visited and n.is_viewable and n_key not in self.frontier and not n.is_occupied:
          self.frontier[n_key] = n
          n.is_frontier = True


  def plan_full_coverage_path(self, plot: bool = True):
    self.coverage_path = [[self.robot.x, self.robot.y, self.robot.yaw]]
    robot_cell = self.grid.get_cell_from_coords(self.robot.x, self.robot.y)
    self.grid.reset_visited()
    self.grid.mark_viewable_cells(robot_cell)

    start_cells = self.grid.get_within_cells(self.robot.body_poly)
    self.maybe_visit_and_update_frontier(start_cells)

    visited_cells = self.grid.get_overlap_cells(self.robot.camera_poly)
    
    for cell in visited_cells:
      view_angle = self.robot.get_viewing_angle(cell.x, cell.y)
      if (self.robot.in_view(view_angle) and not self.grid.is_occluded(robot_cell, cell)):
        self.maybe_visit_and_update_frontier([cell])


    if plot:
      self.plot()


    while len(self.frontier):
      robot_cell = self.grid.get_cell_from_coords(self.robot.x, self.robot.y)
      
      goal_path = None
      goal_cell = None
      goal_distance = np.inf

      for cell in self.frontier.values():
        path, distance = self.get_shortest_path_bfs(robot_cell, cell)

        if goal_distance > distance:
          goal_distance = distance
          goal_cell = cell
          goal_path = path

        break

      next_cell = path[0]

      self.robot.rotate(self.robot.get_angle(cell.x, cell.y))    
      self.robot.translate(next_cell.x, next_cell.y)
      

      visited_cells = self.grid.get_overlap_cells(self.robot.camera_poly)
      
      for cell in visited_cells:
        view_angle = self.robot.get_viewing_angle(cell.x, cell.y)
        if (self.robot.in_view(view_angle) and not self.grid.is_occluded(robot_cell, cell)):
          self.maybe_visit_and_update_frontier([cell])

      self.coverage_path.append([self.robot.x, self.robot.y, self.robot.yaw])

      if plot:
        self.plot()
      break


  def plot(self):
    f = plt.figure(figsize=(20, 20))
    ax = f.add_subplot(1, 1, 1)
    self.grid.plot(ax)
    self.robot.plot(ax)
    plt.show()

