#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String
import tf
import numpy as np
import copy
from numpy import linalg
from utils import wrapToPi
from planners import AStar, compute_smoothed_traj
from grids import StochOccupancyGrid2D
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum
from asl_turtlebot.msg import DetectedObject, DetectedObjectList

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3
    STOP = 4
    CROSS = 5

class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """
    def __init__(self):
        rospy.init_node('turtlebot_navigator', anonymous=True)
        self.mode = Mode.IDLE
        self.lastPlanSuccess = False

        # Current state - todo: woodoo testing
        self.x = 0
        self.y = 0
        self.theta = 0

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None

        self.th_init = 0.0

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0,0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution =  0.1
        self.plan_horizon = 15

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.,0.]
        
        # Robot limits
        self.v_max = 0.2   # maximum velocity
        self.om_max = 0.4   # maximum angular velocity

        self.v_des = 0.1  # desired cruising velocity
        self.theta_start_thresh = 0.1   # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = 0.1     # threshold to be far enough into the plan to recompute it

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.1
        self.at_thresh = 0.02
        self.at_thresh_theta = 0.05

        # trajectory smoothing
        self.spline_alpha = 0.15
        self.traj_dt = 0.05

        # trajectory tracking controller parameters
        self.kpx = 0.5
        self.kpy = 0.5
        self.kdx = 1.5
        self.kdy = 1.5

        # heading controller parameters
        self.kp_th = 2

        # Time to stop at a stop sign
        self.stop_time = 5. #rospy.get_param("~stop_time", 3.)
        self.crossing_time = 30. #rospy.get_param("~crossing_time", 3.)
        self.stopTime = 0
        self.currentStopStartTime = 0
        self.crossTime = 0
        self.currentCrossStartTime = 0


        self.traj_controller = TrajectoryTracker(self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max)
        self.pose_controller = PoseController(0.2, 0.2, 0.2, self.v_max, self.om_max)
        self.heading_controller = HeadingController(self.kp_th, self.om_max)

        self.nav_planned_path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.nav_smoothed_path_pub = rospy.Publisher('/cmd_smoothed_path', Path, queue_size=10)
        self.nav_smoothed_path_rej_pub = rospy.Publisher('/cmd_smoothed_path_rejected', Path, queue_size=10)
        self.nav_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)


        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)

        self.allowUpdateMap = True

        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/map_metadata', MapMetaData, self.map_md_callback)
        rospy.Subscriber('/cmd_nav', Pose2D, self.cmd_nav_callback)
        rospy.Subscriber('/cat', DetectedObject, self.cat_detected_callback)
        #rospy.Subscriber('/finishexplorer', String, self.explorer_map_finish_callback)

        print "finished init"
        
    def dyn_cfg_callback(self, config, level):
        rospy.loginfo("Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}".format(**config))
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]
        return config

    def cat_detected_callback(self, msg):
        if self.currentCrossStartTime != 0 and (rospy.get_rostime() - self.currentCrossStartTime).to_sec() < self.crossing_time:
            return
        dist_s = msg.distance
        if dist_s > 0 and self.mode == Mode.TRACK:
            self.currentStopStartTime = 0
            self.switch_mode(Mode.STOP)

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        new_goal = self.snap_to_grid((data.x, data.y))
        print(new_goal)
        if new_goal[0] != self.x_g or new_goal[1] != self.y_g or data.theta != self.theta_g:
            if (self.mode == Mode.IDLE or self.mode == Mode.PARK):
                rospy.logerr("Replanning to new goal %s, %s, %s", data.x, data.y, data.theta)
                self.x_g = new_goal[0]
                self.y_g = new_goal[1]
                self.theta_g = data.theta
                self.lastPlanSuccess = self.replan(True)
        elif not self.lastPlanSuccess:
            rospy.logerr("Last planning failed, Replanning to new goal %s, %s, %s", data.x, data.y, data.theta)
            self.x_g = new_goal[0]
            self.y_g = new_goal[1]
            self.theta_g = data.theta
            self.lastPlanSuccess = self.replan(True)

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x,msg.origin.position.y)

    def explorer_map_finish_callback(self, msg):
        '''
        After explorer finish, mark space around obstacles unreachable
        '''
    
        self.allowUpdateMap = False
        probs = [p for p in self.map_probs] 

        for i in range(self.map_width):
            for j in range(self.map_height):
                if self.map_probs[i * self.map_width + j] > 90:
                    for k in range(1, 2):
                        probs[(i-k) * self.map_width + j-k] = 100
                        probs[(i-k) * self.map_width + j] = 100
                        probs[(i-k) * self.map_width + j+k] = 100
                        probs[(i) * self.map_width + j-k] = 100
                        probs[(i) * self.map_width + j] = 100
                        probs[(i) * self.map_width + j+k] = 100
                        probs[(i+k) * self.map_width + j-k] = 100
                        probs[(i+k) * self.map_width + j] = 100
                        probs[(i+k) * self.map_width + j+k] = 100

        self.occupancy = StochOccupancyGrid2D(self.map_resolution,
                                                  self.map_width,
                                                  self.map_height,
                                                  self.map_origin[0],
                                                  self.map_origin[1],
                                                  8,
                                                  probs)

    def map_callback(self,msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if self.map_width>0 and self.map_height>0 and len(self.map_probs)>0 and self.allowUpdateMap:
            self.occupancy = StochOccupancyGrid2D(self.map_resolution,
                                                  self.map_width,
                                                  self.map_height,
                                                  self.map_origin[0],
                                                  self.map_origin[1],
                                                  8,
                                                  self.map_probs)
            if self.x_g is not None and (self.mode == Mode.TRACK or self.mode ==Mode.IDLE):
                # if we have a goal to plan to, replan
                rospy.loginfo("replanning because of new map")
                self.replan() # new map, need to replan

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.near_thresh

    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        return (linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.near_thresh and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta)

    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh)
        
    def close_to_plan_start(self):
        return (abs(self.x - self.plan_start[0]) < self.start_pos_thresh and abs(self.y - self.plan_start[1]) < self.start_pos_thresh)

    def snap_to_grid(self, x):
        return (self.plan_resolution*(int(x[0]/self.plan_resolution)+0.5), 
            self.plan_resolution*(int(x[1]/self.plan_resolution)+0.5))

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i,0]
            pose_st.pose.position.y = traj[i,1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        t = self.get_current_plan_time()

        if self.mode == Mode.PARK:
            V, om = self.pose_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.TRACK:
            V, om = self.traj_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(self.x, self.y, self.theta, t)
        else:
            V = 0.
            om = 0.

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime()-self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def replan(self, goal_changed = False):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """
        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo("Navigator: replanning canceled, waiting for occupancy map.")
            self.switch_mode(Mode.IDLE)
            return False

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        problem = AStar(state_min,state_max,x_init,x_goal,self.occupancy,self.plan_resolution)

        # if not problem.is_free((self.x_g, self.y_g)):
        #     rospy.loginfo("Planning failed")
        #     return

        rospy.loginfo("Navigator: computing navigation plan")
        success =  problem.solve()
        if not success:
            rospy.loginfo("Planning failed")
            return False
        rospy.loginfo("Planning Succeeded")

        planned_path = problem.path
        # iter = 0
        # neighbour = [(-0.05, 0), (0.05, 0), (0, 0.05), (0, -0.05), (0, 0)]
        # success = False
        # while success == False and iter < len(neighbour):
        #     rospy.loginfo("Navigator: computing navigation plan")
        #     success =  problem.solve()
        #     if success:
        #         rospy.loginfo("Planning Succeeded")
        #         break
        #     x_goal_n = self.snap_to_grid((self.x_g + neighbour[iter][0], self.y_g + neighbour[iter][1]))
        #     problem = AStar(state_min,state_max,x_init,x_goal_n,self.occupancy,self.plan_resolution)
        #     iter = iter + 1

        # if not success:
        #     rospy.loginfo("Planning failed")
        #     return

        # planned_path = problem.path
        

        # Check whether path is too short
        if len(planned_path) < 4:
            rospy.loginfo("Path too short to track")
            self.switch_mode(Mode.PARK)
            return False

        # Smooth and generate a trajectory
        traj_new, t_new = compute_smoothed_traj(planned_path, self.v_des, self.spline_alpha, self.traj_dt)

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK and False == goal_changed:
            t_remaining_curr = self.current_plan_duration - self.get_current_plan_time()

            # Estimate duration of new trajectory
            th_init_new = traj_new[0,2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err/self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                rospy.loginfo("New plan rejected (longer duration than current plan)")
                self.publish_smoothed_path(traj_new, self.nav_smoothed_path_rej_pub)
                return False

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0,2]
        self.heading_controller.load_goal(self.th_init)

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return False

        rospy.loginfo("Ready to track")
        #self.switch_mode(Mode.TRACK)
        return True

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print e
                pass
            # rospy.loginfo("current state %s %s %s", self.x, self.y, self.theta)
            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                pass
            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.TRACK:
                if self.near_goal():
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (rospy.get_rostime() - self.current_plan_start_time).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan() # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.STOP:
                print("I found a cat!")
                if self.currentStopStartTime == 0:
                    self.currentStopStartTime = rospy.get_rostime()
                #self.stopTime = self.stopTime + 1
                if (rospy.get_rostime() - self.currentStopStartTime).to_sec() > self.stop_time:
                    self.currentCrossStartTime = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.PARK:
                try:
                    if self.at_goal():
                        # forget about goal:
                        self.x_g = None
                        self.y_g = None
                        self.theta_g = None
                        self.switch_mode(Mode.IDLE)
                except:
                    rospy.logerr("TypeError in at_goal")
                    print(self.x)
                    print(self.y)
                    print(self.x_g)
                    print(self.y_g)
                    print(self.theta)
                    print(self.theta_g)

            self.publish_control()
            rate.sleep()

if __name__ == '__main__':    
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    nav.run()
