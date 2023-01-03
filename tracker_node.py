#!/usr/bin/env python3
"""
ROS2 node for multi object tracking
Author: Zhihao, Christian

NOTE Use this to see the msg types:
https://gitlab.com/autowarefoundation/autoware.auto/autoware_auto_msgs/-/tree/7bba204c7f9ad8b6159f1a63aa679cc66b8839a1/autoware_auto_msgs/msg

Input: AutowareDetectedObjects:

Subscribe to:            "/lidar/detected_objects"

Input data structure:

 autoware_auto_msgs.msg.DetectedObject(
     existence_probability=1.0, 
     classification=[autoware_auto_msgs.msg.ObjectClassification(
         classification=0, probability=1.0)], 
     kinematics=autoware_auto_msgs.msg.DetectedObjectKinematics(
         centroid_position=geometry_msgs.msg.Point(x=-6.117222785949707, y=-4.258943557739258, z=0.6000003814697266), 
         position_covariance=array([0., 0., 0., 0., 0., 0., 0., 0., 0.]), 
         has_position_covariance=False, 
         orientation=geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0), 
         orientation_availability=0, 
         twist=geometry_msgs.msg.TwistWithCovariance(
             twist=geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0),
             angular=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0)), 
             covariance=array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), 
             has_twist=False, 
             has_twist_covariance=False), 
     shape=autoware_auto_msgs.msg.Shape(
        polygon=geometry_msgs.msg.Polygon(
            points=[
                geometry_msgs.msg.Point32(x=-4.200207710266113, y=-3.1997604370117188, z=-0.2999992370605469), 
                geometry_msgs.msg.Point32(x=-7.87706184387207, y=-2.955186605453491, z=-0.2999992370605469), 
                geometry_msgs.msg.Point32(x=-8.0342378616333, y=-5.318126201629639, z=-0.2999992370605469), 
                geometry_msgs.msg.Point32(x=-4.357383728027344, y=-5.562700271606445, z=-0.2999992370605469)]), 
        height=1.7999992370605469))


Output: (Autoware) TrackedObjects

Publish to:                  "/perception/tracked_objects"

msg: autoware_auto_msgs.msg.TrackedObjects(
    header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=0, nanosec=0), frame_id=''),
    objects=[object,object])

    object: autoware_auto_msgs.msg.TrackedObject(
        object_id=0,
        existence_probability=1.0,
        classification=[autoware_auto_msgs.msg.ObjectClassification(
         classification=0, probability=1.0)],
        kinematics=autoware_auto_msgs.msg.TrackedObjectKinematics( 
                centroid_position=geometry_msgs.msg.Point(x=-6.117222785949707, y=-4.258943557739258, z=0.6000003814697266), 
                position_covariance=array([0., 0., 0., 0., 0., 0., 0., 0., 0.]), 
                has_position_covariance=False, 
                orientation=geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0), 
                orientation_availability=0, 
                twist=geometry_msgs.msg.TwistWithCovariance(
                    twist=geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0),
                    angular=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0)), 
                    covariance=array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), 
                    has_twist=False, 
                    has_twist_covariance=False), 
        shape=autoware_auto_msgs.msg.Shape(
            polygon=geometry_msgs.msg.Polygon(
                points=[
                    geometry_msgs.msg.Point32(x=-4.200207710266113, y=-3.1997604370117188, z=-0.2999992370605469), 
                    geometry_msgs.msg.Point32(x=-7.87706184387207, y=-2.955186605453491, z=-0.2999992370605469), 
                    geometry_msgs.msg.Point32(x=-8.0342378616333, y=-5.318126201629639, z=-0.2999992370605469), 
                    geometry_msgs.msg.Point32(x=-4.357383728027344, y=-5.562700271606445, z=-0.2999992370605469)]), 
                height=1.7999992370605469))


TODO : The plan for next step:
1. Tune the parameters for the tracker
2. Take the parameters outside the tracker into a yaml file

These steps are not necessary:
3. use a 3D object motion model inside the tracker

"""
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Header
from geometry_msgs.msg import Polygon, Point, Point32, Pose, Quaternion, Twist, TwistWithCovariance, Vector3
import numpy as np
from autoware_auto_msgs.msg import DetectedObjects, TrackedObjects, TrackedObject, ObjectClassification, \
    Shape, ObjectClassification, TrackedObjectKinematics

from .tracker import MultiTracker, Track

# from tf2_ros import TransformException
# from tf2_ros.buffer import Buffer
# from tf2_ros.transform_listener import TransformListener


class ArtTracker2Node(Node):
    def __init__(self):
        super().__init__('art_tracker2')
        self.subscription = self.create_subscription(DetectedObjects, 'detected_objects', \
                self.callback_1, rclpy.qos.qos_profile_sensor_data)

        self.publisher_ = self.create_publisher(TrackedObjects, 'tracked_objects', rclpy.qos.qos_profile_sensor_data)
        # self.publisher_viz = self.create_publisher(Point, '/perception/tracked_objects_viz', 10)
        self.frame_id = 'base_link'
        self.dt = 0.1 # We can hard code that the lidar stack is at 10Hz, but later we use the time stamp from the lidar
        self.tracker = MultiTracker()
        self.last_timestamp = None

        #setup the transform
        # self.tf_buffer = Buffer()
        # self.tf_listener = TransformListener(self.tf_buffer, self)
        # self.target_frame = 'map'

    def callback_1(self, data):
        '''
        observations -> [[x,y],[x,y],...] is a 2D array
        state_estimate -> [[x,y,vx,vy],[x,y,vx,vy],...]
        trackedObjects -> [TrackedObject,TrackedObject,...]
        '''

        # get the dt from the message time stamps
        if self.last_timestamp is None:
            self.last_timestamp = data.header.stamp.sec + data.header.stamp.nanosec/1e9
            return
        else:
            this_stamp = data.header.stamp.sec + data.header.stamp.nanosec/1e9
            self.dt = this_stamp - self.last_timestamp
            self.last_timestamp = this_stamp

        # 1. transform the data to the observation numpy array
        observations = []
        
        # self.get_logger().info('data: {}'.format(data.objects[0]))

        # This is for filter out objects which are too high or too low
        for detected_obj in data.objects:
            x = detected_obj.kinematics.centroid_position.x
            y = detected_obj.kinematics.centroid_position.y
            z = detected_obj.kinematics.centroid_position.z
            if z > 0 and z < 2.5:
                observations.append([x, y])

        
        # The ob should then be [[x,y],[x,y],...]
        # self.get_logger().info('observations: {}'.format(observations[:3]))

        # 2. pass the observation to the tracking algorithm using updateTracker(observations, dt, obsCov=None)
        objects_dict = self.tracker.updateTracker(observations, self.dt) # dt = 0.1 
        

        # 3.Assimble the TrackedObjects message

            # assimble the header
        self.msg_header = Header()
        self.msg_header.stamp = self.get_clock().now().to_msg()
        self.msg_header.frame_id = self.frame_id #This is the reference frame for the pose data.
            # self.msg_header.seq = data.header.seq
        self.assembled_msg = TrackedObjects()
        self.assembled_msg.header = self.msg_header

        for id, object in objects_dict.items():
            #  get the tracked Track objects, use getState() to get the state estimate
            state = object.getState()

            #slice the covariance matrix to get only the 2x2 matrix for position
            stateCovariance = object.getStateCovariance()[:2,:2] 

            #pad the covariance matrix from 2x2 to 3x3
            stateCovariance = np.pad(stateCovariance, ((0,1),(0,1)), 'constant', constant_values=0.0)

            

            object_msg = TrackedObject()

            object_msg.object_id = id
            object_msg.existence_probability = 1.0
            classifi = ObjectClassification()
            classifi.classification = 0
            classifi.probability = 1.0
            object_msg.classification = [classifi]
            object_msg.kinematics=TrackedObjectKinematics( 
                centroid_position= Point(x=state[0], y=state[1], z=0.0), 
                position_covariance=list( stateCovariance.flatten()), 
                # has_position_covariance=False, 
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0), 
                orientation_availability=0, 
                twist=TwistWithCovariance(
                    twist=Twist(linear=Vector3(x=0.0, y=0.0, z=0.0),
                    angular=Vector3(x=0.0, y=0.0, z=0.0)), 
                    covariance=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0.]), ) 
            object_msg.shape=[Shape(
                polygon=Polygon(
                    points=[Point32(x=state[0]+1, y=state[1]+1, z=-0.3), 
                        Point32(x=state[0]-1, y=state[1]+1, z=-0.3), 
                        Point32(x=state[0]-1, y=state[1]-1, z=-0.3), 
                        Point32(x=state[0]+1, y=state[1]-1, z=-0.3)]), 
                    height=1.7)]
            # self.get_logger().info('flatted: {}'.format(list( stateCovariance.flatten())))

            # self.get_logger().info('msg: {}'.format(self.assembled_msg))
            self.assembled_msg.objects.append(object_msg)
        # 4. publish the state estimate to the /perception/tracked_objects topic

        self.publisher_.publish(self.assembled_msg)
    
def main():
    rclpy.init()
    node = ArtTracker2Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
