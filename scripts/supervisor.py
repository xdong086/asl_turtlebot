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
    STOP = 8
    CROSS = 9

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
        self.pos_eps = rospy.get_param("~pos_eps", 0.2)
        self.theta_eps = rospy.get_param("~theta_eps", 0.5)        


        if verbose:
            print("SupervisorParams:")
            print("    use_gazebo = {}".format(self.use_gazebo))
            '''
            print("    rviz = {}".format(self.rviz))
            '''
            print("    mapping = {}".format(self.mapping))
            print("    pos_eps, theta_eps = {}, {}".format(self.pos_eps, self.theta_eps))

class Supervisor:

    def __init__(self):
        # Initialize ROS node
        rospy.init_node('turtlebot_supervisor', anonymous=True)
        self.params = SupervisorParams(verbose=True)

        # Food delivery queue
        self.itemQueue = Queue.Queue()
        self.orderQueue = Queue.Queue()
        self.last_order_id = "id"

        # PICUP mode timer
        self.pickupTimer = 0

        # Current state
        self.x = 3.15
        self.y = 1.6
        self.theta = 0
        
        # Food object names
        self.valid_food_names = {"hot_dog": None, "apple": None, "donut": None}
        self.go_to_food_locations = {}
        self.last_updated = None
        # For testing phase 2, delete when doing demo
        #self.valid_food_names = {"hot_dog": (2.2851, -0.0211), "apple": (0.0804, 0.0722), "donut": (2.061, 2.1991)}
        #self.valid_food_names = {"hot_dog": (3.15, 1.0), "apple": (0.4, 0.4), "donut": (2.061, 2.1991)}
        #self.go_to_food_locations = {"hot_dog": (3.39, 2.78), "apple": (3.35, 0.30), "donut": (3.35, 0.30)}

        # Explore waypoints list
        self.explore_waypoints = [(3.39, 2.78, 1.62), (0.61, 2.77, -3.12), (0.32, 2.22, -2.08), (0.25, 1.65, -2.08), 
                                   (0.25, 0.37, -0.06), (2.27, 0.33, -3.0), (2.30, 1.62, 0), (2.27, 0.4, -2.0), 
                                   (3.35, 0.30, 1.63), (3.09, 1.38, -1.56)]
        #self.explore_waypoints = []
        self.next_waypoint_index = -1
        # Goal state
        #self.x_g, self.y_g, self.theta_g = self.explore_waypoints[self.next_waypoint_index]
        self.x_g = self.y_g = self.theta_g = None

        # Current mode
        self.mode = Mode.IDLE
        #self.switch_mode(Mode.WAIT) #Mode.EXPLORE
        self.switch_mode(Mode.EXPLORE)

        self.tfBroadcaster = tf2_ros.TransformBroadcaster()

        ########## PUBLISHERS ##########

        # Command pose for controller
        self.pose_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)

        # Command vel (used for idling)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Food object marker
        self.food_viz_pub = rospy.Publisher("/viz/food_marker", Marker, queue_size=10)

        # Explorer finish
        self.explorer_finish_pub = rospy.Publisher("/finishexplorer", String, queue_size=10)

        self.cat_pub = rospy.Publisher('/cat', DetectedObject, queue_size=10)
        ########## SUBSCRIBERS ##########

        # Listen to order
        rospy.Subscriber('/delivery_request', String , self.order_callback)

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
        items = msg.data.split(",")
        if items[0] == self.last_order_id:
            return
        else:
            self.last_order_id = items[0]
            self.orderQueue.put(msg)

    def fill_item_queue(self, order):
        items = order.data.split(",")
        seen = []
        for i in items[1:]:
            if i not in seen and i in self.go_to_food_locations:
                print("will get",i)
                self.itemQueue.put(i)
                seen.append(i)
        """
        for i in set(items[1:]):
            if i in self.go_to_food_locations:
                self.itemQueue.put(i)
        """

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

        self.switch_mode(Mode.EXPLORE)

    def nav_pose_callback(self, msg):
        self.x_g = msg.x
        self.y_g = msg.y
        self.theta_g = msg.theta
        self.switch_mode(Mode.EXPLORE)

    def object_detected_callback(self, msg):
        for ob_msg in msg.ob_msgs:
            #rospy.loginfo("%s object detected", ob_msg.name)
            if ob_msg.name == "cat":
                self.cat_pub.publish(ob_msg)
            # TODO
            if ob_msg.name in self.valid_food_names and \
                ob_msg.distance < 1.5 and \
               (ob_msg.name not in self.go_to_food_locations or \
                self.last_updated == ob_msg.name):
                food_heading = np.arctan2(np.sin(ob_msg.thetaleft) + np.sin(ob_msg.thetaright), np.cos(ob_msg.thetaleft) + np.cos(ob_msg.thetaright))
                food_heading_x = self.x + ob_msg.distance * np.cos(food_heading + self.theta)
                food_heading_y = self.y + ob_msg.distance * np.sin(food_heading + self.theta)
                dest_x = self.x + max(max(ob_msg.distance, 0.0) * np.cos(food_heading + self.theta) - 0.2, 0)
                dest_y = self.y + max(max(ob_msg.distance, 0.0) * np.sin(food_heading + self.theta) - 0.2, 0)
                rospy.loginfo("%s food: %s, %s, %s", ob_msg.name, ob_msg.distance, food_heading_x, food_heading_y)
                rospy.loginfo("%s food head: %s, %s, %s", ob_msg.name, food_heading, ob_msg.thetaleft, ob_msg.thetaright)
                self.valid_food_names[ob_msg.name] = (food_heading_x, food_heading_y)
                self.go_to_food_locations[ob_msg.name] = (dest_x, dest_y)
                self.last_updated = ob_msg.name

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
        #rospy.loginfo("%s, %s, %s, %s ,%s, %s", x, self.x, y, self.y, theta, self.theta)
        deltaTheta = abs(theta - self.theta)
        if deltaTheta > 3.14:
            deltaTheta = 2*3.14 - deltaTheta
        
        return abs(x - self.x) < self.params.pos_eps and \
               abs(y - self.y) < self.params.pos_eps and \
               deltaTheta < self.params.theta_eps
    ########## Code ends here ##########


    ########## STATE MACHINE LOOP ##########
    def switch_mode(self, new_mode):
        rospy.loginfo(">>>>>SUPERVISOR: Switching from %s -> %s", self.mode, new_mode)
        self.prev_mode = new_mode
        self.mode = new_mode

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
        rospy.loginfo("Current mode: %s, %s, %s", self.mode, self.x_g, self.y_g)
        ########## Code starts here ##########
        #print("----------- %s"%(self.mode))
        #print("----------- %d"%(self.itemQueue.qsize()))
        if self.mode == Mode.IDLE:
            self.stay_idle()

        elif self.mode == Mode.EXPLORE:
            if self.next_waypoint_index == -1 or self.close_to(self.x_g, self.y_g, self.theta_g):
                self.next_waypoint_index += 1
                if self.next_waypoint_index < len(self.explore_waypoints):
                    self.x_g, self.y_g, self.theta_g = self.explore_waypoints[self.next_waypoint_index]
                    self.nav_to_pose()
                else:
                    self.switch_mode(Mode.WAIT)
            else:
                self.nav_to_pose()
        elif self.mode == Mode.WAIT:
            if self.orderQueue.empty():
                self.stay_wait()
            else:
                order = self.orderQueue.get()
                self.fill_item_queue(order)
                item = self.itemQueue.get()
                self.x_g, self.y_g = self.go_to_food_locations[item]
                self.theta_g = 0
                self.switch_mode(Mode.NAVTOVENDER)
        elif self.mode == Mode.NAVTOVENDER:
            if self.close_to(self.x_g, self.y_g, self.theta_g):
                self.switch_mode(Mode.PICKUP)
            else:
                self.nav_to_pose()
        elif self.mode == Mode.PICKUP:
            # wait for 3 seconds
            if self.pickupTimer == 0:
                self.pickupTimer = rospy.get_rostime()
            #self.pickupTimer = self.pickupTimer + 1
            if (rospy.get_rostime()-self.pickupTimer).to_sec() > 10:
                if self.itemQueue.empty():
                    self.x_g, self.y_g, self.theta_g = (3.09, 1.38, -1.56)
                    self.pickupTimer = 0
                    self.switch_mode(Mode.NAVTOCUSTOMER)
                else:
                    item = self.itemQueue.get()
                    self.x_g, self.y_g = self.go_to_food_locations[item]
                    self.theta_g = 0
                    self.pickupTimer = 0
                    self.switch_mode(Mode.NAVTOVENDER)
        elif self.mode == Mode.NAVTOCUSTOMER:
            if self.close_to(self.x_g, self.y_g, self.theta_g):
                self.switch_mode(Mode.WAIT)
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
        """

        for i, food_name in enumerate(self.go_to_food_locations):
            if self.go_to_food_locations[food_name]:
                self.publish_food_marker("@"+food_name[0], i, self.go_to_food_locations[food_name])        
        """
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
