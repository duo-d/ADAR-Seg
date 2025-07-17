import rospy
import utm
from geometry_msgs.msg import Twist,PoseStamped, Vector3
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
import math
import yaml
import numpy as np
import pandas as pd
from RRT import find_path as rrt
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import glob
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation
try:
    with open(r'config/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
except:
    print("YAML loading error!...")
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from  control_system.srv import Traversability, TraversabilityRequest


p_alcanzado = False
field_names = "['x', 'y', 'z']"

#INITIAL REFERENCE
x_ref,y_ref, _, _ = utm.from_latlon(config['latitud_ref'],config['longitud_ref'])
#INITIALIZE VARIABLES for Lyapunov control
i,alpha_prev,d_prev = 0 , 0.0 , 0.0
x_ant, y_ant, v_ant = -0.01 , -0.01 ,0
o_act = 0 

#ROSBAG
class process_data():
    def __init__(self):
        self.idx_pc = [] #point cloud index
        self.idx_tf_odom = [] #matrix transforms between odom and base_link
        self.idx_base = [] #sampled waypoints
        self.data=[]
        self.root= '/media/arvc/Extreme SSD/rosbag-video/def/2024-07-23-17-58-19/robot0/lidar/data/'
        self.static = '/media/arvc/Extreme SSD/rosbag-video/def/2024-07-23-17-58-19/robot0/tf_static/data.csv'
        self.root_tf_dinamic='/media/arvc/Extreme SSD/rosbag-video/def/2024-07-23-17-58-19/robot0/tf/data.csv'
        self.t=0
        
    def takeClosest(self,num,collection):
        return min(collection,key=lambda x:abs(x-num))

    def find_nearest(self,idx,timestamp_pc,timestamp):
        idx_2 = []
        for i in idx:
            idx_2.append(self.takeClosest(timestamp[i],timestamp_pc))
        return idx_2

    def get_pc_idx(self,idx,timestamp_gps):
        df_pc = pd.read_csv('/media/arvc/Extreme SSD/rosbag-video/def/2024-07-23-17-58-19/robot0/lidar/data.csv', dtype = 'str' )
        timestamp_pc = df_pc['#timestamp [ns]'].to_numpy(dtype=np.uint64)
        return self.find_nearest(idx,timestamp_pc,timestamp_gps)

    def get_basetf_idx(self,idx,timestamp_gps):
        df_tf = pd.read_csv(self.root_tf_dinamic, dtype = 'str' )
        timestamp_tf = df_tf['timestamp'].to_numpy(dtype=np.uint64)
        idx_tf_odom=self.find_nearest(idx,timestamp_tf,timestamp_gps)
        mask=np.isin(timestamp_tf,idx_tf_odom)
        idx_3=np.where(mask==True)[0]
        return idx_3

    def get_idx_gps(self):

        df_base = pd.read_csv('/media/arvc/Extreme SSD/rosbag-video/def/2024-07-23-17-58-19/robot0/odom/data.csv', dtype = 'str' )
        self.timestamp_gps = df_base['#timestamp [ns]'].to_numpy(dtype=np.uint64)
        all_x = df_base['x'].to_numpy(dtype=np.float32)
        all_y = df_base['y'].to_numpy(dtype=np.float32)
        #sample waypoints every X meters
        x = all_x[0]
        y = all_y[0]
        x_prev = x
        y_prev = y
        self.idx_base.append(0)
        for i in range(1,len(all_x)):
            x = all_x[i]
            y = all_y[i]
            dist = math.sqrt((x-x_prev)**2 + (y-y_prev)**2)
            if dist > 5:
                x_prev = x
                y_prev = y
                self.idx_base.append(i)
                self.data.append([x,y])
        return np.array(self.data), self.idx_base
    
    def sample(self):
        self.data,self.idx_base = self.get_idx_gps()
        self.idx_pc = self.get_pc_idx(self.idx_base,self.timestamp_gps)
        self.idx_tf_odom = self.get_basetf_idx(self.idx_base,self.timestamp_gps)
        return self.data,self.idx_base,self.idx_pc,self.idx_tf_odom
    
    def leer_coordenadas(self): 
        self.data,self.idx_base,self.idx_pc,self.idx_tf_odom = self.sample()
        self.x2 = self.data[:,0]
        self.y2 = self.data[:,1]

    def call_traversability_service(self,cloud):
        print("Calling the traversability analysis service...")
        rospy.wait_for_service('traversability_analysis')
        try:
            traversability_service = rospy.ServiceProxy('traversability_analysis', Traversability)
            req = TraversabilityRequest(input=cloud)
            resp = traversability_service(req)
            return resp.trav_points, resp.no_trav_points
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def read_tf(self,root,idx): #0,8,12
        df_tf = pd.read_csv(root, dtype='a')
        row=df_tf.loc[idx][:]
        t=np.array([row["x"],row["y"],row["z"],]).astype(np.float128)
        rot = Rotation.from_quat([row["qx"],row["qy"],row["qz"],row["qw"]])
        euler = rot.as_euler('xyz', degrees=False)
        tf = np.array([[np.cos(euler[2]), -np.sin(euler[2]), 0, 0],
                    [np.sin(euler[2]), np.cos(euler[2]), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
        tf[:3,3]=t
        return tf

    def transform_pc(self):
        tf_static_lidar_plate=self.read_tf(self.static,4)
        tf_static_plate_base=self.read_tf(self.static,1)
        tf_dinamic_base_odom=self.read_tf(self.root_tf_dinamic,self.idx_tf_odom[self.t])
        multiple_transformations = tf_dinamic_base_odom @ tf_static_plate_base @ tf_static_lidar_plate 
        return multiple_transformations

    def leer_points(self):
        print("Reading point cloud...", str(self.idx_pc[self.t]))
        print(self.root+str(self.idx_pc[self.t])+'.pcd')
        nube = o3d.io.read_point_cloud(self.root+str(self.idx_pc[self.t])+'.pcd')
        self.transformation_matrix=self.transform_pc()
        nube_tf=nube.transform(self.transformation_matrix)
        voxel_size = 0.02  # Adjust the voxel size as needed
        downsampled_pcd = nube_tf.voxel_down_sample(voxel_size)
        points = np.asarray(downsampled_pcd.points)
        #remove points that are too close to the robot
        # p_removed=np.where((np.linalg.norm(points, axis=1)<2))[0]
        # new_points=np.delete(points,p_removed, axis=0)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "base_link"
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1), # 4
                    PointField('z', 8, PointField.FLOAT32, 1),  # 8
                    #PointField('rgb', 16, PointField.UINT32, 1),
                    ]
        nube_ros=pc2.create_cloud(header,fields,points)

        if self.t < len(self.idx_pc):
            self.t = self.t+1
            print("T =", self.t)
        else:
            print("Fin pointclouds")
        return nube_ros


    def points(self):
        # Service call    
        points = self.leer_points()
        trav, no_trav = self.call_traversability_service(points) 

        trav = list(pc2.read_points(trav, skip_nans=True, field_names=field_names))
        trav_points = np.asarray(trav)
        print("Traversable points",trav_points.shape)
        no_trav = list(pc2.read_points(no_trav, skip_nans=True, field_names=field_names))
        no_trav_points = np.asarray(no_trav)
        print("Non-traversable points",no_trav_points.shape)
        pc_trav = o3d.geometry.PointCloud()
        pc_obs = o3d.geometry.PointCloud()
        pc_trav.points = o3d.utility.Vector3dVector(np.asarray(trav_points))
        pc_obs.points = o3d.utility.Vector3dVector(np.asarray(no_trav_points))
        tranversable = np.asarray(pc_trav.points)
        obstacles = np.asarray(pc_obs.points)
        return tranversable, obstacles

        
    def create_marker(self,marker_id,a,b):
        #Draw the trajectory in geo-referenced map
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "multi_marker"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Define the position of the marker
        marker.pose.position.x = a
        marker.pose.position.y = b
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set the scale of the marker (1x1x1 here means 1 meter in each dimension)
        marker.scale.x = 0.7
        marker.scale.y = 0.7
        marker.scale.z = 0.7

        # Set the color of the marker
        marker.color.r = 0.5
        marker.color.g = 0.0
        marker.color.b = 0.5
        marker.color.a = 1.0
        return marker
    




def global_path(x2,y2):

    path_global.header.stamp = rospy.Time.now()
    path_global.header.frame_id = "odom"  

    # Add points to the trajectory
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "odom"
    pose.pose.position.x = x2
    pose.pose.position.y = y2
    pose.pose.position.z = 0
    path_global.poses.append(pose)

    path_pub.publish(path_global)    

def talker(x, z):
    pub.publish(Twist(linear=Vector3(x=x), angular=Vector3(z=z)))

def publish_path(x,y):

    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = "odom"  

    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "odom"
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = 0
    path_msg.poses.append(pose)
    path_publisher.publish(path_msg)



def control(x2,y2):
    global x_ant,y_ant,alpha_prev,d_prev,v_ant,i,o_act,p_alcanzado
    x,y,z = my_position()
    publish_path(x,y)


    #Distance to the goal and orientation
    d_desplazado = math.hypot(x - x_ant, y - y_ant)  #Distancia euclidea
    d_destino = math.hypot(x2[0] - x, y2[0] - y)     #Distancia hasta el pto destino
    o_des = math.atan2((y2[0]-y),(x2[0]-x))  #*(180/math.math.pi)

    if d_desplazado > 0.1 :
        o_act = math.atan2((y-y_ant),(x-x_ant))   
        
    
    #ANGULAR ERROR
    o_act , o_des = (o_act + 2*math.pi)%(2*math.pi) , (o_des + 2*math.pi)%(2*math.pi)  #[-pi,pi] --> [0,2pi]
    alpha = o_des - o_act  
    alpha = ((alpha+math.pi) % (2*math.pi) ) - math.pi   #[0,2pi] --> [-pi,pi]

    #LINEAR ERROR
    d = math.hypot(x2[0] - x, y2[0] - y)  

    #CONTROL 
    v = config['kd']*(d)*math.cos(alpha)
    w = config['kd']*math.cos(alpha)*math.sin(alpha) + config['kang']*alpha 

    #Limit speed and rotation
    v = np.clip(v,config['vmin'],config['vmax'])
    w = np.clip(w,config['wmin'],config['wmax'])
    #Publish control
    talker(v,w)

    alpha_prev, d_prev, v_ant = alpha , d , v
    if d_desplazado > 0.1 :  
        x_ant,y_ant = x,y
    
    # LOOP for reached point
    if  d_destino < config['precision'] : 
        print('\nPUNTO ALCANZADO\n')
        if(len(x2) == 1):  
            p_alcanzado = True
        else:
            x2 = x2[1:]
            y2 = y2[1:] 
            global_path(x2[0],y2[0])
            print("x3:",x2[0],"y3",y2[0])
    
    return x2,y2
   
class get_pose_current():
    def __init__(self):
        self.latitud  = 0.0
        self.longitud = 0.0
        self.message_received = False

    def listener(self):
        self.sub=rospy.Subscriber('/navsat/fix' , NavSatFix, self.callback_pos)

    def callback_pos(self, data):
        self.latitud = data.latitude
        self.longitud = data.longitude
        self.message_received = True

def my_position():
    x_i,y_i, _, _ = utm.from_latlon(pose.latitud, pose.longitud)
    x,y  = x_i - x_ref , y_i - y_ref            #Local coordinates
    z = 0
    return x,y,z


if __name__ == '__main__':

    rospy.init_node('control',anonymous=True)
    r = rospy.Rate(10)  

    pub = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist , queue_size=10)
    path_publisher = rospy.Publisher('/husky/path', Path, queue_size=10)    
    path_pub = rospy.Publisher('/global_path', Path, queue_size=10)    
    marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
    path_msg = Path()
    path_global = Path()
    marker_array = MarkerArray()

    pose = get_pose_current()
    pose.listener()
    process_data = process_data()
    

    
    print("Waiting gps...")
    while not pose.message_received:
        pass

    process_data.leer_coordenadas()
    for i, pos in enumerate(process_data.data):
        marker = process_data.create_marker(i, pos[0], pos[1])
        marker_array.markers.append(marker)
    
    print("Waiting traversability analysis service...")
    aux=0

    while not rospy.is_shutdown():
        for marker in marker_array.markers:
            marker_pub.publish(marker_array)
        start = np.asarray(my_position())
        goal = [process_data.x2[0],process_data.y2[0],0]
       
        tranversable,obstacles = process_data.points()
        #coloring the point cloud
        traversable_pc = o3d.geometry.PointCloud()
        color=[126,135,237]
        traversable_pc.points = o3d.utility.Vector3dVector(tranversable)
        traversable_pc.paint_uniform_color(np.asarray(color).astype(float)/255)
        #unify in one point cloud
        obstacles_pc = o3d.geometry.PointCloud()
        color=[148,210,168] 
        obstacles_pc.points = o3d.utility.Vector3dVector(obstacles)
        obstacles_pc.paint_uniform_color(np.asarray(color).astype(float)/255)
        final_pcd=traversable_pc+obstacles_pc

        print("MY POSITION:",start, "GOAL:",goal)
        x3,y3,reached,colision, path_smooth = rrt(start,goal,tranversable,obstacles,aux)
        aux=aux+1

        if colision or x3 is None:
            process_data.x2,process_data.y2 = process_data.x2[1:],process_data.y2[1:]  #Me salto este objetivo

        else:
            while not p_alcanzado:
                x3,y3 = control(x3,y3) 
            
            p_alcanzado = False
            process_data.x2,process_data.y2 = process_data.x2[1:],process_data.y2[1:] #Removed reached point
            print("x2:",process_data.x2[0],"y2",process_data.y2[0])

    r.sleep()
