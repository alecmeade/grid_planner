import argparse

from frontier_planner import FrontierPlanner
from occupancy_grid import OccupancyGrid
from point_cloud import read_point_cloud_log, PointCloud2D, PointCloud3D 
from robot import Robot

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--pcl_path', type=str, default = "pcl/sas_pc_0.bin")
    arg_parser.add_argument('--grid_resolution', type=int, default = 0.8)
    arg_parser.add_argument('--z_min', type=float, default = 0.5)
    arg_parser.add_argument('--z_max', type=float, default = 0.6)
    arg_parser.add_argument('--start_x', type=int, default = 0)
    arg_parser.add_argument('--start_y', type=int, default = 0)
    arg_parser.add_argument('--robot_radius', type=float, default = 0.8)
    arg_parser.add_argument('--robot_yaw', type=float, default = 0)
    arg_parser.add_argument('--robot_sensor_range', type=float, default = 7)
    arg_parser.add_argument('--robot_sensor_view_angle', type=float, default = 60)
    arg_parser.add_argument('--wall_obseration_angle', type=float, default = 70)
    arg_parser.add_argument('--output', type=str, default = "waypoints.txt")
    arg_parser.add_argument('--show_plot', type=bool, default = True)
    args = arg_parser.parse_args()

    # Transform the provided 3D point cloud to a 2D grid representation.
    pcl_3d = PointCloud3D(read_point_cloud_log("pcl/sas_pc_0.bin", 3, False))
    pcl_2d = pcl_3d.to_2D(args.z_min, args.z_max)
    map_grid = OccupancyGrid(pcl_2d, args.grid_resolution)

    # Create a robot within the grid world.
    robot = Robot(args.start_x, args.start_y, 
                  args.robot_radius, 
                  args.robot_yaw,
                  args.robot_sensor_range, 
                  args.robot_sensor_view_angle)

    # Plan a full coverage path for the robot to explore the gridworld.
    planner = FrontierPlanner(map_grid, robot, args.wall_obseration_angle, args.show_plot)
    waypoints = planner.get_full_coverage_path()

    # Write the planned waypoints out to a file for later use.
    with open(args.output, "w") as f:
        for p in waypoints:
            f.write(",".join([str(v) for v in p]))
            f.write("\n")

