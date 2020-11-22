import copy
import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
import time
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
    self.rotation_interpolation_step = 10
    self.top_n_frontier = 5


  def astar(self, start: OccupancyCell, goal: OccupancyCell) -> Tuple[List[OccupancyCell], float]:
    if self.is_visible(start, goal):
      return [start], [start.x_axis_angle(goal)], 0

    distances = {}
    path_history = {}
    path_history[start.key()] = None
    current_cell = None
    min_heap = []
    heapq.heappush(min_heap, [0, 0, 0, start])

    while len(min_heap) and current_cell is None:
      _, _, _, cell = heapq.heappop(min_heap)
      for n in self.grid.get_neighbors(cell): 
        if n.key() in path_history:
          continue

        path_history[n.key()] = cell

        if (not self.is_collision(n)):
          d = cell.euclidean_distance(n)
          distances[n.key()] = d

          if d <= self.robot.sensor_range and self.is_visible(n, goal):
            current_cell = n
            break

          heapq.heappush(min_heap, (goal.euclidean_distance(n), utils.angular_distance(n.x_axis_angle(goal), self.robot.yaw), np.random.rand(), n))

    path = [current_cell]
    angles = [current_cell.x_axis_angle(goal)]
    distance = 0

    while current_cell.key() in path_history and path_history[current_cell.key()] is not None:
      path.insert(0, current_cell)
      prev_cell = path_history[current_cell.key()]
      heading = prev_cell.x_axis_angle(current_cell)
      distance += distances[current_cell.key()]
      current_cell = prev_cell

    return path, angles, distance


  def bfs(self, start: OccupancyCell, goal: OccupancyCell) -> Tuple[List[OccupancyCell], float]:
    if self.is_visible(start, goal):
      return [start], [start.x_axis_angle(goal)], 0

    distances = {}
    path_history = {}
    path_history[start.key()] = None
    current_cell = None
    queue = [start]

    while len(queue) and current_cell is None:
      cell = queue.pop(0)
      for n in self.grid.get_neighbors(cell): 
        if n.key() in path_history:
          continue

        path_history[n.key()] = cell

        if ((n.is_visited and not n.is_occupied) or not self.is_collision(n)):
          d = cell.euclidean_distance(n)
          distances[n.key()] = d

          if d <= self.robot.sensor_range and self.is_visible(n, goal):
            current_cell = n
            break

          queue.append(n)


    path = [current_cell]
    angles = [current_cell.x_axis_angle(goal)]
    distance = 0

    while current_cell.key() in path_history and path_history[current_cell.key()] is not None:
      path.insert(0, current_cell)
      prev_cell = path_history[current_cell.key()]
      heading = prev_cell.x_axis_angle(current_cell)
      distance += distances[current_cell.key()]
      current_cell = prev_cell

    return path, angles, distance


  def normals_in_range(self, current: OccupancyCell, goal: OccupancyCell, normal_range: float = 70) -> float:
    neighbors = self.grid.get_neighbors(goal)
    point_slopes = [] 
    distances = []

    min_angle = 360
    for (x1, y1), (x2, y2), (row_offset, col_offset) in goal.get_sides():
      neighbor = self.grid.get_cell(goal.row + row_offset, goal.col + col_offset)
      if neighbor is None:
        continue

      x3 = (x1 + x2) / 2.0
      y3 = (y1 + y2) / 2.0

      if not neighbor.is_occupied and neighbor.is_viewable:
        v1 = (current.x - goal.x, current.y - goal.y)
        v2 = (x3 - goal.x, y3 - goal.y)
        dot_product = np.dot(v1 / np.linalg.norm(v1), 
                             v2 / np.linalg.norm(v2))
        angle = math.degrees(np.arccos(dot_product))
        min_angle = min(min_angle, angle)

    return normal_range >= min_angle


  def is_visible(self, current: OccupancyCell, goal: OccupancyCell) -> bool:
    if not goal.is_viewable:
      return False

    new_robo = copy.deepcopy(self.robot) 
    x_offset = current.x - self.robot.x
    y_offset = current.y - self.robot.y
    new_robo.translate(current.x, current.y)
    new_robo.rotate(current.x_axis_angle(goal))
 
    if ((new_robo.camera_poly.overlaps(goal.poly) or goal.poly.within(new_robo.camera_poly)) and
        not self.grid.is_occluded(current, goal) and 
        new_robo.in_view(goal.x, goal.y) and
        self.normals_in_range(current, goal)):
      return True

    return False


  def is_collision(self, cell: OccupancyCell) -> bool:
    if cell.is_visited and cell.is_occupied:
      return True

    new_body_poly = geometry.Point(cell.x, cell.y).buffer(self.robot.radius)
    cells = self.grid.get_overlap_cells(new_body_poly)

    for cell in cells:
      if (cell.is_visited and cell.is_occupied):
        return True

    return False


  def maybe_visit_and_update_frontier(self, cells: List[OccupancyCell]):
    for cell in cells:
      if cell.is_visited or not cell.is_viewable:
        continue

      cell.is_visited = True
      cell.is_frontier = False
      self.frontier.pop(cell.key(), None)

      for n in self.grid.get_neighbors(cell, include_diagonals = True):
        n_key = n.key()
        if (not n.is_visited and 
            n.is_viewable and 
            n_key not in self.frontier):
          self.frontier[n_key] = n
          n.is_frontier = True


  def observe_cells(self, view_cell, camera_poly):
      for cell in self.grid.get_overlap_cells(camera_poly):
        if (not cell.is_visited and self.is_visible(view_cell, cell)):                 
          self.maybe_visit_and_update_frontier([cell])


  def plan_full_coverage_path(self, plot: bool = True):
    self.coverage_path = [[self.robot.x, self.robot.y, self.robot.yaw]]
    self.grid.reset_visited()
    robot_cell = self.grid.get_cell_from_coords(self.robot.x, self.robot.y)
    
    self.grid.mark_viewable_cells(robot_cell)

    start_cells = self.grid.get_overlap_cells(self.robot.body_poly)
    self.maybe_visit_and_update_frontier(start_cells)
    self.observe_cells(robot_cell, self.robot.camera_poly)


    if plot:
      self.plot()


    robot_cell = self.grid.get_cell_from_coords(self.robot.x, self.robot.y)

    while len(self.frontier):
      goal_path = None
      goal_cell = None
      goal_distance = np.inf
      goal_rotation = 360

      frontier_distances = []
      for cell in self.frontier.values():
        frontier_distances.append([cell.key(), robot_cell.euclidean_distance(cell)])

      frontier_distances = sorted(frontier_distances, key = lambda x: x[1])
      n_frontier = min(len(frontier_distances), self.top_n_frontier)

      for i in range(0, n_frontier):        
        cell = self.frontier[frontier_distances[i][0]]        
        path, angles, distance = self.astar(robot_cell, cell)
        rotation = utils.angular_distance(angles[0], self.robot.yaw)

        if (goal_distance > distance) or (goal_distance >= distance and goal_rotation > rotation):
          goal_rotation = rotation
          goal_distance = distance
          goal_cell = cell
          goal_path = path


      next_cell = goal_path[0]
      current_heading = self.robot.yaw
      new_heading = None


      if next_cell.is_visited and next_cell != robot_cell:
        self.robot.translate(next_cell.x, next_cell.y)
        current_heading = self.robot.yaw
        self.observe_cells(robot_cell, self.robot.camera_poly)
        
        if plot:
          self.plot()

      new_heading = utils.wrap_heading(current_heading, self.robot.get_x_axis_angle(goal_cell.x, goal_cell.y))

      if next_cell != robot_cell:
        new_heading = utils.wrap_heading(current_heading, current_heading + ((new_heading - current_heading) / (len(goal_path) - 1)))


      robot_cell = self.grid.get_cell_from_coords(self.robot.x, self.robot.y)

      self.coverage_path.append([self.robot.x, self.robot.y, self.robot.yaw])

      for r in np.linspace(current_heading, new_heading, int(abs(current_heading - new_heading) / self.rotation_interpolation_step)):
        self.robot.rotate(r)  
        self.observe_cells(robot_cell, self.robot.camera_poly)
        if plot:
          self.plot()

      self.observe_cells(robot_cell, self.robot.camera_poly)
      self.robot.rotate(new_heading)  

      if plot:
        self.plot()


  def plot(self):
    f = plt.figure(figsize=(6, 6))
    ax = f.add_subplot(1, 1, 1)
    self.grid.plot(ax)
    self.robot.plot(ax)
    if len(self.coverage_path) > 1:
      for i in range(1, len(self.coverage_path)):
        x1, y1, _ = self.coverage_path[i - 1]
        x2, y2, _ = self.coverage_path[i]
        ax.plot([x1, x2], [y1, y2], color='k')
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()
    # plt.show()
