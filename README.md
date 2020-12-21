# Frontier Based Grid Planner

The code in this repo is an implementation of 2D grid planner. A 3D point cloud is discretized to a 2D occupancy
grid representation and then a frontier based planner using A* hueristics is used to explore the occupancy grid. Instead,
of typical grid world representations that visit cells by occupying them, this implementation creates a robot and camera sensor
which are directed to observe the space. Cosntraints are placed on occlusions and viewing angles between occupied cells in the 
environment to count them as visited. The work in this directory was completed for a term project in Fall 2020 MIT 16.485,
Visual Navigation for Autonomous Vehicles. 

## Demo

A video demo of the grid planner code running can be found here.
https://drive.google.com/file/d/1P-0FvcL454sZbnoOzVRJzHNoTNxpMzBT/view?usp=sharing


## Writeup

The writeup of the term project and results that this code was used in can be found here.
https://drive.google.com/file/d/1AbfM8XJMGBp9qH_0qX-y5AOLkJcosEns/view?usp=sharing
