import torch
import torch.nn as nn
from torch.optim import SGD
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
from tenext import TeNeXt
import time
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
import struct

from control_system.srv import Traversability, TraversabilityResponse  
import struct

class TraversabilityService:
    def __init__(self):
        self.device = torch.device('cpu')
        self.root = "TeNeXt/BestModel9_th_0.4862739620804787voxel_size0.2_0.9197958031905511.pth"
        self.model = TENext(1, 1).to(self.device)
        self.model.load_state_dict(torch.load(self.root, map_location=torch.device('cpu')))
        self.criterion = nn.BCELoss()
        self.optimizer = SGD(self.model.parameters(), lr=1e-1)
        self.threshold = 0.4862739620804787
        self.voxel_size = 0.2

        self.pub = rospy.Publisher("trav_analysis", PointCloud2, queue_size=2)

    def handle_traversability(self, req):
        start = time.time()
        field_names = [field.name for field in req.input.fields]
        points = list(pc2.read_points(req.input, skip_nans=True, field_names=field_names))

        if len(points) == 0:
            rospy.logwarn("Received an empty cloud")
            return TraversabilityResponse()
        else:
            pcd_array = np.asarray(points)
            pointcloud = o3d.geometry.PointCloud()
            pointcloud.points = o3d.utility.Vector3dVector(pcd_array[:, 0:3])
            coords_orig = np.asarray(pointcloud.points)
            coords = ME.utils.batched_coordinates([coords_orig / self.voxel_size], dtype=torch.float32)
            self.features = np.ones((coords_orig.shape[0], 1))

            # Inference
            test_in_field = ME.TensorField(torch.from_numpy(self.features).to(dtype=torch.float32),
                                           coordinates=coords,
                                           quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                           minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                                           device=self.device)
            test_output = self.model(test_in_field.sparse())
            logit = test_output.slice(test_in_field)
            pred_raw = logit.F.detach().cpu().numpy()
            pred = np.where(pred_raw > self.threshold, 1, 0)

            points = self.visualize_each_cloud(pred, coords_orig)
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "base_link"
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1),
                      PointField('rgb', 16, PointField.UINT32, 1)]

            # Built cloud and publish it
            self.pc2_trav=pc2.create_cloud(header,fields,points)
            self.pub.publish(self.pc2_trav)
            # Separate traversable and non-traversable points
            transversable_points = [pt for pt, pred_label in zip(points, pred) if pred_label == 1]
            no_transversable_points = [pt for pt, pred_label in zip(points, pred) if pred_label == 0]

            # Create PointCloud2 messages
            transversable_points_msg = pc2.create_cloud(header, fields, transversable_points)
            no_transversable_points_msg = pc2.create_cloud(header, fields, no_transversable_points)

            stop = time.time()
            duration = stop - start
            rospy.loginfo(f"Processing duration: {duration}")

            return TraversabilityResponse(trav_points=transversable_points_msg,
                                          no_trav_points=no_transversable_points_msg)

    def visualize_each_cloud(self, pred, coords):
        points = []
        for k, i in enumerate(pred):
            if i == 1:
                r, g, b, a = 95, 158, 160, 255
            else:
                r, g, b, a = 106, 90, 205, 255
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            pt = [coords[k, 0], coords[k, 1], coords[k, 2], rgb]
            points.append(pt)
        return points

if __name__ == '__main__':
    rospy.init_node('traversability_service')
    service = TraversabilityService()
    rospy.Service('traversability_analysis', Traversability, service.handle_traversability)
    rospy.loginfo('Traversability service ready.')
    rospy.spin()