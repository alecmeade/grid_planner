import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import utils
from shapely import affinity, geometry


class Robot():

    def __init__(self, x: float, y: float, radius: float, yaw: float, sensor_range: float, sensor_view_angle: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.yaw = yaw
        self.sensor_range = sensor_range
        self.sensor_view_angle = sensor_view_angle
        self.sensor_width =  self.sensor_range * (math.tan(math.radians(self.sensor_view_angle / 2.0))) * 2

        self.yaw_rad = math.radians(yaw)
        self.camera_poly = geometry.Polygon([[self.x, self.y],
                    [self.x + self.sensor_range * math.cos(self.yaw_rad) - self.sensor_width * math.sin(self.yaw_rad) / 2,
                     self.y + self.sensor_range * math.sin(self.yaw_rad) + self.sensor_width * math.cos(self.yaw_rad) / 2],
                    [self.x + self.sensor_range * math.cos(self.yaw_rad) + self.sensor_width * math.sin(self.yaw_rad) / 2,
                     self.y + self.sensor_range * math.sin(self.yaw_rad) - self.sensor_width * math.cos(self.yaw_rad) / 2]])
        self.body_poly = geometry.Point(x, y).buffer(radius)


    def plot(self, ax):
        rect = patches.Circle((self.x, self.y),
                               self.radius,
                               linewidth = 2,
                               color = 'r',
                               linestyle = 'dashed',
                               fill = False,
                               alpha = 1)
        ax.add_patch(rect)
        ax.arrow(self.x, 
                 self.y,
                 (self.radius * math.cos(self.yaw_rad)), 
                 (self.radius * math.sin(self.yaw_rad)),
                 length_includes_head = True,
                 linewidth = 2)

        camera_poly_coords = np.asarray(self.camera_poly.exterior.coords)
        tri = plt.Polygon(camera_poly_coords, 
                          fill = False, 
                          color ='r', 
                          linestyle = 'dashed', 
                          alpha = 1, 
                          linewidth = 2)
        ax.add_patch(tri)


    def rotate(self, yaw: float):
        yaw = yaw % 360
        yaw_offset = yaw - self.yaw
        self.yaw += yaw_offset
        self.yaw_rad = math.radians(yaw)
        self.body_poly = affinity.rotate(self.body_poly, yaw_offset, origin='centroid')
        self.camera_poly = affinity.rotate(self.camera_poly, yaw_offset, (self.x, self.y))


    def translate(self, x: float, y: float):
        x_offset = x - self.x
        y_offset = y - self.y
        self.x += x_offset
        self.y += y_offset
        self.body_poly = affinity.translate(self.body_poly, xoff = x_offset, yoff = y_offset)
        self.camera_poly = affinity.translate(self.camera_poly, xoff = x_offset, yoff = y_offset)


    def get_z_axis_angle(self, x: float, y: float) -> float:
        return utils.z_axis_angle(self.x, self.y, x, y)


    def get_view_offset_angle(self, x: float, y) -> float:
        x_axis_angle = self.get_z_axis_angle(x, y)
        x_axis_angle = utils.wrap_heading(self.yaw, x_axis_angle)
        return x_axis_angle - self.yaw


    def print(self, prefix: str = "Robot"):
        print("%s | X: %0.2f | Y: %0.2f | YAW: %0.2f" % (prefix, self.x, self.y, self.yaw))


    def in_view(self, x: float, y: float) -> bool:
        if abs(self.get_view_offset_angle(x, y)) < (self.sensor_view_angle / 2.0):
            return True

        return False

