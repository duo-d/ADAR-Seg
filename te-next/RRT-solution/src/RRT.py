import matplotlib.pyplot as plt

from RRTPlanner.rrtplanner.rrtplannerPC import RRTPlannerPC
from RRTPlanner.rrtplanner.trajectorysmoother import TrajectorySmoother

import rospy
from sensor_msgs.msg import PointCloud2
import time


def find_path(start,goal,tranversable,obstacles,idx):
    start_time = time.time()    
    planner = RRTPlannerPC(start=start,
                           goal=goal,
                           pc_traversable=tranversable,
                           pc_obstacles=obstacles,
                           epsilon=0.13,
                           robot_radius=0.55,
                           max_nodes=2000)

    
    if planner.goal_colision():
        print("Goal collision detected. Aborting path planning.")
        return None, None, False, True
    else:
        tree = planner.build_rrt_connect_to_goal()
        planner.print_info()
        # retrieve found path path from tree
        path_found = planner.get_solution_path(tree)
        # print('Optimal path: ', path_found)
        
        # optionally, find a smooth path  
        smoother = TrajectorySmoother(s=1)
        path_smooth = smoother.smooth3D(path_found) #Error m > k must hold --> revisar ruta path_found
        # path_smooth[:,2] = -0.7
        # path_smooth = path_found
        # print('Smoothed path: ', path_smooth)

        stop = time.time()
        duration = stop - start_time
        
        # plot the tree and solution
        tree.plot(label = 'Tree')
        tree.plot_path(path=path_found,  color='yellow', label='Solution Path', linewidth=2)
        tree.plot_path(path=path_smooth, color='cyan', label='Smooth Path', linewidth=3)
           
        # plot obstacles, start and goal
        # planner.plot()
        # plt.legend(loc='upper right')
        # plt.legend([])
        # update plots
        # plt.tight_layout()
        # plt.axis([start[0]-10, start[0]+10, start[1]-10, start[1]+10])
        
        # plt.show(block=True)
        #save the plot as png
        # plt.savefig('/home/arvc/Vídeos/videosQ1/capturas2D/'+str(idx)+'.png')
        # plt.close()

        print('FINISHED')
        rospy.loginfo(f"Processing duration: {duration}")

        x3 = path_smooth[:,0]
        y3 = path_smooth[:,1]

        return x3,y3,planner.goal_reached, False, path_smooth

# if __name__ == "__main__":
#     rospy.init_node('rrt_sub',anonymous=True)

#     r = rospy.Rate(10)
#     start = [0, 0, 0]
#     goal = [2, 5 , 0]
#     points_traversable = points_traversable()
#     points_traversable.listener()

#     points_obstacles = points_obstacles()
#     points_obstacles.listener()

#     while not rospy.is_shutdown():
#         if not points_traversable.message_received and not points_obstacles.message_received:
#             print("No points")
#         else:
#             find_path_RRT_connect_to_goal_point_cloud(start,goal)
#     r.sleep()