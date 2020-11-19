#!/usr/bin/env python

from enum import Enum

import rospy
from asl_turtlebot.msg import DetectedObject, DetectedObjectList
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from std_msgs.msg import Float32MultiArray, String
from visualization_msgs.msg import Marker
from localization import create_transform_msg
import tf
import tf2_ros
import numpy as np
import Queue

class Mode(Enum):
    """State machine modes. Feel free to change."""
    EXPLORE = 1
    IDLE = 2 
    WAIT = 3
    NAVTOVENDER = 4
    PICKUP = 5
    NAVTOCUSTOMER = 6
    FINISHONE = 7

class SupervisorParams:

    def __init__(self, verbose=False):
        # If sim is True (i.e. using gazebo), we want to subscribe to
        # /gazebo/model_states. Otherwise, we will use a TF lookup.
        self.use_gazebo = rospy.get_param("sim")

        '''
        # How is nav_cmd being decided -- human manually setting it, or rviz
        self.rviz = rospy.get_param("rviz")
        '''

        # If using gmapping, we will have a map frame. Otherwise, it will be odom frame.
        self.mapping = rospy.get_param("map")

        # Threshold at which we consider the robot at a location
        self.pos_eps = rospy.get_param("~pos_eps", 0.15)
        self.theta_eps = rospy.get_param("~theta_eps", 0.35)

        # Time to stop at a stop sign
        self.stop_time = rospy.get_param("~stop_time", 3.)

        # Minimum distance from a stop sign to obey it
        self.stop_min_dist = rospy.get_param("~stop_min_dist", 0.5)

        # Time taken to cross an intersection
        self.crossing_time = rospy.get_param("~crossing_time", 3.)

        if verbose:
            print("SupervisorParams:")
            print("    use_gazebo = {}".format(self.use_gazebo))
            '''
            print("    rviz = {}".format(self.rviz))
            '''
            print("    mapping = {}".format(self.mapping))
            print("    pos_eps, theta_eps = {}, {}".format(self.pos_eps, self.theta_eps))
            print("    stop_time, stop_min_dist, crossing_time = {}, {}, {}".format(self.stop_time, self.stop_min_dist, self.crossing_time))


class Supervisor:

    def __init__(self):
        # Initialize ROS node
        rospy.init_node('turtlebot_supervisor', anonymous=True)
        self.params = SupervisorParams(verbose=True)

        # Food delivery queue
        self.orderQueue = Queue.Queue()

        # PICUP mode timer
        self.pickupTimer = 0

        # Current state - todo: woodoo testing
        self.x = 3.15
        self.y = 1.6
        self.theta = 0
        
        # Food object names
        self.valid_food_names = {"hot_dog": None, "apple": None, "donut": None}

        # For testing phase 2, delete when doing demo
        #self.valid_food_names = {"hot_dog": (2.2851, -0.0211), "apple": (0.0804, 0.0722), "donut": (2.061, 2.1991)}

        # Explore waypoints list
        self.explore_waypoints = [(3.39, 2.78, 1.62), (0.66, 2.77, -3.12), (0.32, 2.22, -2.08), (0.29, 1.65, -2.08), 
                                  (0.31, 0.37, -0.06), (2.27, 0.33, -3.0), (2.30, 1.62, 0), (2.27, 0.4, -2.0), 
                                  (3.35, 0.30, 1.63), (3.09, 1.38, -1.56)]
        self.next_waypoint_index = 0
        # Goal state
        self.x_g, self.y_g, self.theta_g = self.explore_waypoints[self.next_waypoint_index]

        # Current mode
        self.mode = Mode.EXPLORE
        self.prev_mode = None  # For printing purposes

        self.tfBroadcaster = tf2_ros.TransformBroadcaster()

        ########## PUBLISHERS ##########

        # Command pose for controller
        self.pose_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)

        # Command vel (used for idling)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Food object marker
        self.food_viz_pub = rospy.Publisher("/viz/food_marker", Marker, queue_size=10)
        ########## SUBSCRIBERS ##########

        # Listen to order
        rospy.Subscriber('/order', String , self.order_callback)

        # Object detector
        rospy.Subscriber('/detector/objects', DetectedObjectList, self.object_detected_callback)

        # High-level navigation pose
        rospy.Subscriber('/nav_pose', Pose2D, self.nav_pose_callback)

        # If using gazebo, we have access to perfect state
        if self.params.use_gazebo:
            rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_callback)
        self.trans_listener = tf.TransformListener()

        '''
        # If using rviz, we can subscribe to nav goal click
        if self.params.rviz:
            rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)
        else:
            self.x_g, self.y_g, self.theta_g = 1.5, -4., 0.
            self.mode = Mode.NAV
        '''

    ########## SUBSCRIBER CALLBACKS ##########

    def order_callback(self, msg):
        self.orderQueue.put(msg.data)

    def gazebo_callback(self, msg):
        if "turtlebot3_burger" not in msg.name:
            return

        pose = msg.pose[msg.name.index("turtlebot3_burger")]
        self.x = pose.position.x
        self.y = pose.position.y
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]

    def rviz_goal_callback(self, msg):
        """ callback for a pose goal sent through rviz """
        origin_frame = "/map" if self.params.mapping else "/odom"
        print("Rviz command received!")

        try:
            nav_pose_origin = self.trans_listener.transformPose(origin_frame, msg)
            self.x_g = nav_pose_origin.pose.position.x
            self.y_g = nav_pose_origin.pose.position.y
            quaternion = (nav_pose_origin.pose.orientation.x,
                          nav_pose_origin.pose.orientation.y,
                          nav_pose_origin.pose.orientation.z,
                          nav_pose_origin.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            self.theta_g = euler[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

        self.mode = Mode.EXPLORE

    def nav_pose_callback(self, msg):
        self.x_g = msg.x
        self.y_g = msg.y
        self.theta_g = msg.theta
        self.mode = Mode.EXPLORE

    def object_detected_callback(self, msg):
        for ob_msg in msg.ob_msgs:
            rospy.loginfo("%s object detected", ob_msg.name)
            if ob_msg.name in self.valid_food_names:
                food_heading = np.arctan2(np.sin(ob_msg.thetaleft) + np.sin(ob_msg.thetaright), np.cos(ob_msg.thetaleft) + np.cos(ob_msg.thetaright))
                food_heading_x = self.x + ob_msg.distance * np.cos(food_heading + self.theta)
                food_heading_y = self.y + ob_msg.distance * np.sin(food_heading + self.theta)
                rospy.loginfo("%s food: %s, %s, %s", ob_msg.name, ob_msg.distance, food_heading_x, food_heading_y)
                rospy.loginfo("%s food head: %s, %s, %s", ob_msg.name, food_heading, ob_msg.thetaleft, ob_msg.thetaright)
                self.valid_food_names[ob_msg.name] = (food_heading_x, food_heading_y)
    ########## STATE MACHINE ACTIONS ##########

    ########## Code starts here ##########
    # Feel free to change the code here. You may or may not find these functions
    # useful. There is no single "correct implementation".

    def go_to_pose(self):
        """ sends the current desired pose to the pose controller """

        pose_g_msg = Pose2D()
        pose_g_msg.x = self.x_g
        pose_g_msg.y = self.y_g
        pose_g_msg.theta = self.theta_g

        self.pose_goal_publisher.publish(pose_g_msg)

    def nav_to_pose(self):
        """ sends the current desired pose to the naviagtor """

        nav_g_msg = Pose2D()
        nav_g_msg.x = self.x_g
        nav_g_msg.y = self.y_g
        nav_g_msg.theta = self.theta_g

        self.pose_goal_publisher.publish(nav_g_msg)

    def stay_idle(self):
        """ sends zero velocity to stay put """

        vel_g_msg = Twist()
        self.cmd_vel_publisher.publish(vel_g_msg)

    def stay_wait(self):
        """ sends zero velocity to stay put """

        vel_g_msg = Twist()
        self.cmd_vel_publisher.publish(vel_g_msg)

    def close_to(self, x, y, theta):
        """ checks if the robot is at a pose within some threshold """

        return abs(x - self.x) < self.params.pos_eps and \
               abs(y - self.y) < self.params.pos_eps and \
               abs(theta - self.theta) < self.params.theta_eps

    ########## Code ends here ##########


    ########## STATE MACHINE LOOP ##########

    def loop(self):
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """
        if not self.params.use_gazebo:
            try:
                origin_frame = "/map" if self.params.mapping else "/odom"
                translation, rotation = self.trans_listener.lookupTransform(origin_frame, '/base_footprint', rospy.Time(0))
                self.x, self.y = translation[0], translation[1]
                self.theta = tf.transformations.euler_from_quaternion(rotation)[2]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass

        # logs the current mode
        if self.prev_mode != self.mode:
            rospy.loginfo("Current mode: %s", self.mode)
            self.prev_mode = self.mode

        ########## Code starts here ##########
        # TODO: Currently the state machine will just go to the pose without stopping
        #       at the stop sign.
        if self.mode == Mode.IDLE:
            # Send zero velocity
            self.stay_idle()
        elif self.mode == Mode.EXPLORE:
            if self.close_to(self.x_g, self.y_g, self.theta_g):
                self.next_waypoint_index += 1
            
            if self.next_waypoint_index == len(self.explore_waypoints):
                self.mode = Mode.WAIT
            else:
                self.x_g, self.y_g, self.theta_g = self.explore_waypoints[self.next_waypoint_index]
                self.nav_to_pose()
        elif self.mode == Mode.WAIT:
            if self.orderQueue.empty():
                self.stay_wait()
            else:
                order = self.orderQueue.get()
                self.x_g, self.y_g = self.valid_food_names[order]
                self.theta_g = 0
                self.mode = Mode.NAVTOVENDER
        elif self.mode == Mode.NAVTOVENDER:
            if self.close_to(self.x_g, self.y_g, self.theta_g):
                self.mode = Mode.PICKUP
            else:
                self.nav_to_pose()
        elif self.mode == Mode.PICKUP:
            # wait for 3 seconds
            self.pickupTimer = self.pickupTimer + 1
            if self.pickupTimer > 30:
                self.x_g, self.y_g, self.theta_g = (3.09, 1.38, -1.56)
                self.mode = Mode.NAVTOCUSTOMER
        elif self.mode == Mode.NAVTOCUSTOMER:
            if self.close_to(self.x_g, self.y_g, self.theta_g):
                self.mode = Mode.WAIT
            else:
                self.nav_to_pose()
        #elif self.node == Mode.FINISHONE:
        else:
            raise Exception("This mode is not supported: {}".format(str(self.mode)))


        '''
        Publish food markers
        '''
        for i, food_name in enumerate(self.valid_food_names):
            if self.valid_food_names[food_name]:
                self.publish_food_marker(food_name, i, self.valid_food_names[food_name])
        ############ Code ends here ############

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()

    def publish_food_marker(self, food_name, id, pos):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = id
        marker.type = marker.TEXT_VIEW_FACING
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = 0.4
        marker.scale.x = .15
        marker.scale.y = .15
        marker.scale.z = .15
        marker.color.a = 1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.text = food_name
        self.food_viz_pub.publish(marker) 

if __name__ == '__main__':
    sup = Supervisor()
    sup.run()
