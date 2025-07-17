#!/usr/bin/env python
# encoding: utf-8
"""
Find a path on a point cloud.
Each point in the pointcloud is classified as traversable or non-traversable.
We are looking for a 3D trajectory using RRT, with the following characteristics:
    - The robot (sphere), should have traversable points, ensuring that there exists a surface to navigate on.
    - In addition the robot (sphere), should not have non-traversable points, considered as obstacles.
    - An RRT Planning with multiple goals considered.
    - A holonomic robot is considered.

- Under development:
    - Find a smooth trajectory.
    - Integrate RRT CONNECT.
    - Integrate max delta(phi) between nodes in the tree.
    - Integrate max delta(Z) between the points.

    - Probabilistic roadmap.

Two slight modifications are performed over the stantard RRT:
    a) The algorithm looks for a path to any of the goals specified.
    b) The algorithm stops when a path is found for any of the goals.
    c) The algorithm returns, always, as a solution:
        b.1) The paths that connects the start and one of the goals.
        b.2) If no path was found, the algorithm finds the closest node to any of the goals and returns the path that
        connects the nearest neighbour node with the start node.

In the RRTPlannerMG a solution is always found by the planner, either  by reaching any of the goals or
by finding the closest node to any of the goals. This, of course, is a desired behaviour, since the robot
has always a movement plan to carry out.


@Authors: Arturo Gil
@Time: April 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from keyframe.keyframe import KeyFrame
from rrtplanner.rrtplannerPC import RRTPlannerPC
from rrtplanner.trajectorysmoother import TrajectorySmoother

from mpl_toolkits.mplot3d import Axes3D

def find_path_RRT_basic_point_cloud():
    smoother = TrajectorySmoother(s=5)
    # transformation from the robot center (at the ground) to the LIDAR.
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0.8],
                  [0, 0, 0, 1]])
    # simulation: must subscribe to lidar and compute traversability
    pcl = KeyFrame()
    pcl.from_file('dataset/robot0/lidar/001.pcd')
    pcl.downsample(voxel_size=0.2)
    pcl.transform(T)
    # pointcloud.draw_cloud()
    # a simple classification in traversable/non-traversable if any points surpasses Z
    points_traversable, points_obstacles = pcl.compute_traversability(Z=0.1)
    
    # start and goals
    start = [0, 0, 0]
    # goals should be projected to the local reference system from the global UTM coordinates
    goal = [3, 2, 0]
    # Create Multiple Goal RRT planner
    planner = RRTPlannerPC(start=start,
                           goal=goal,
                           pc_traversable=points_traversable,
                           pc_obstacles=points_obstacles,
                           robot_radius=0.7,
                           epsilon=0.5,
                           max_nodes=3500)

    tree = planner.build_rrt_basic()
    planner.print_info()
    # retrieve found path path from tree
    path_found = planner.get_solution_path(tree)
    print('Optimal path: ', path_found)

    # optionally, find a smooth path
    # path_smooth = smoother.smooth3D(path_found)
    path_smooth = path_found
    print('Smoothed path: ', path_smooth)

    # plot the tree and solution
    tree.plot()
    tree.plot_path(path=path_found, color='cyan')
    tree.plot_path(path=path_smooth, color='green')
    # plot obstacles, start and goal
    planner.plot()

    # update plots
    plt.show(block=True)
    print('FINISHED')


def find_path_RRT_connect_point_cloud():
    smoother = TrajectorySmoother(s=5)
    # transformation from the robot center (at the ground) to the LIDAR.
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0.8],
                  [0, 0, 0, 1]])
    # simulation: must subscribe to lidar and compute traversability
    pcl = KeyFrame()
    pcl.from_file('dataset/robot0/lidar/001.pcd')
    pcl.downsample(voxel_size=0.2)
    pcl.transform(T)
    # pointcloud.draw_cloud()
    # a simple classification in traversable/non-traversable if any points surpasses Z
    points_traversable, points_obstacles = pcl.compute_traversability(Z=0.1)

    # start and goals
    start = [0, 0, 0]
    # goals should be projected to the local reference system from the global UTM coordinates
    goal = [16, 16 , 0]
    # goal = [-8, -8.5, 0]
    # Create Multiple Goal RRT planner
    planner = RRTPlannerPC(start=start,
                           goal=goal,
                           pc_traversable=points_traversable,
                           pc_obstacles=points_obstacles,
                           robot_radius=0.7,
                           epsilon=0.5,
                           max_nodes=3500)

    # returns two trees that may be connected
    treeA, treeB = planner.build_rrt_connect()
    planner.print_info()
    # retrieve found paths from trees
    path_found = planner.get_solution_path_from_two_trees(treeA, treeB)
    print('Optimal path: ', path_found)

    # optionally, find a smooth path
    path_smooth = smoother.smooth3D(path_found)
    print('Smoothed path: ', path_smooth)

    # plot the tree and solution
    treeA.plot()
    treeB.plot()
    treeA.plot_path(path=path_found, color='cyan')
    treeA.plot_path(path=path_smooth, color='green')
    # plot obstacles, start and goal
    planner.plot()

    # update plots
    plt.show(block=True)
    print('FINISHED')


def find_path_RRT_connect_to_goal_point_cloud():
    smoother = TrajectorySmoother(s=5)
    # transformation from the robot center (at the ground) to the LIDAR.
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0.8],
                  [0, 0, 0, 1]])
    # simulation: must subscribe to lidar and compute traversability
    pcl = KeyFrame()
    pcl.from_file('dataset/robot0/lidar/pruebacoppelia.pcd')
    # pcl.from_points(np.asarray(pcl.pointcloud.points))
    pcl.downsample(voxel_size=0.5)

    # pcl.draw_cloud()
    pcl.draw_cloud_plt()

    pcl.transform(T)

    pcl.draw_cloud_plt()

    # a simple classification in traversable/non-traversable if any points surpasses Z
    points_traversable, points_obstacles = pcl.compute_traversability(Z=1) #0.15
    # start and goals
    start = [0, 0, 0]
    # goals should be projected to the local reference system from the global UTM coordinates
    # goal = [5, 0, 0]
    goal = [8, 11, 0]
    # Create Multiple Goal RRT planner
    planner = RRTPlannerPC(start=start,
                           goal=goal,
                           pc_traversable=points_traversable,
                           pc_obstacles=points_obstacles,
                           epsilon=0.7, #0.7
                           robot_radius=0.7,
                           max_nodes=3500)

    # returns two trees that may be connected
    tree = planner.build_rrt_connect_to_goal()
    planner.print_info()
    # retrieve found path path from tree
    path_found = planner.get_solution_path(tree)
    print('Optimal path: ', path_found)

    # optionally, find a smooth path
    path_smooth = smoother.smooth3D(path_found)
    print('Smoothed path: ', path_smooth)

    # plot the tree and solution
    tree.plot()
    tree.plot_path(path=path_found, color='cyan')
    tree.plot_path(path=path_smooth, color='green')
    # plot obstacles, start and goal
    planner.plot()

    # update plots
    plt.show(block=True)
    print('FINISHED')



if __name__ == "__main__":
    # find_path_RRT_basic_point_cloud()
    find_path_RRT_connect_point_cloud()
    find_path_RRT_connect_to_goal_point_cloud()


# 1. `find_path_RRT_basic_point_cloud()`: Este método implementa la versión básica del algoritmo RRT. En cada iteración, selecciona un punto aleatorio en el espacio y trata de conectarlo al árbol existente. Este método puede ser eficiente en términos de tiempo de ejecución, pero la ruta resultante puede no ser la más corta o la más suave.

# 2. `find_path_RRT_connect_point_cloud()`: Este método implementa la variante RRT-Connect del algoritmo RRT. En cada iteración, después de seleccionar un punto aleatorio y conectarlo al árbol existente, también intenta conectar directamente este nuevo punto al objetivo. Esto puede resultar en rutas más directas al objetivo y un tiempo de ejecución más rápido en comparación con el RRT básico.

# 3. `find_path_RRT_connect_to_goal_point_cloud()`: Este método parece ser una variante del RRT-Connect donde se intenta conectar directamente al objetivo más a menudo. Esto podría resultar en rutas aún más directas y un tiempo de ejecución más rápido, pero también podría causar problemas si hay muchos obstáculos en la línea directa hacia el objetivo.

