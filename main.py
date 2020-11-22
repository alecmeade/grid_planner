import argparse

from frontier_planner import FrontierPlanner
from occupancy_grid import OccupancyGrid
from point_cloud import read_point_cloud_log, PointCloud2D, PointCloud3D 
from robot import Robot

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--pcl_path', type=str, default = "pcl/sas_pc_0.bin")
    arg_parser.add_argument('--grid_resolution', type=int, default = 1)
    arg_parser.add_argument('--z_min', type=float, default = 0.5)
    arg_parser.add_argument('--z_max', type=float, default = 0.6)
    arg_parser.add_argument('--start_row', type=int, default = 12)
    arg_parser.add_argument('--start_col', type=int, default = 7)
    arg_parser.add_argument('--robot_radius', type=float, default = 0.6)
    arg_parser.add_argument('--robot_yaw', type=float, default = 0)
    arg_parser.add_argument('--robot_sensor_range', type=float, default = 4)
    arg_parser.add_argument('--robot_sensor_view_angle', type=float, default = 60)
    arg_parser.add_argument('--output', type=str, default = "waypoints.txt")
    args = arg_parser.parse_args()

    pcl_3d = PointCloud3D(read_point_cloud_log("pcl/sas_pc_0.bin", 3, False))
    pcl_2d = pcl_3d.to_2D(args.z_min, args.z_max)
    map_grid = OccupancyGrid(pcl_2d, args.grid_resolution)

    cell = map_grid.get_cell(args.start_row, args.start_col)
    robot = Robot(cell.x, cell.y, 
                  args.robot_radius, 
                  args.robot_yaw,
                  args.robot_sensor_range, 
                  args.robot_sensor_view_angle)

    planner = FrontierPlanner(map_grid, robot)
    waypoints = planner.get_full_coverage_path(plot=True)

    with open(args.output, "w") as f:
        for p in waypoints:
            f.write(",".join([str(v) for v in p]))
            f.write("\n")
