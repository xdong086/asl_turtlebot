import numpy as np
import rospy
from std_msgs.msg import Float64
from utils import wrapToPi

"""Publisher from section 5 added at the end"""
# command zero velocities once we are this close to the goal
RHO_THRES = 0.05
ALPHA_THRES = 0.1
DELTA_THRES = 0.1

class PoseController:

    """ Pose stabilization controller """
    def __init__(self, k1, k2, k3, V_max=0.5, om_max=1):
        print("pose init")
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max
        self.pub_alpha = rospy.Publisher('/controller/alpha', Float64, queue_size=10)
        self.pub_delta = rospy.Publisher('/controller/delta', Float64, queue_size=10)
        self.pub_rho = rospy.Publisher('/controller/rho', Float64, queue_size=10)

    def load_goal(self, x_g, y_g, th_g):
        """ Loads in a new goal position """
        print("load goal")
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g


    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs: 
            V, om: Control actions

        Hints: You'll need to use the wrapToPi function. The np.sinc function
        may also be useful, look up its documentation
        """
        ########## Code starts here ##########

        # dx = self.x_g - x
        # dy = self.y_g - y
        # phi = np.sqrt(dx**2 + dy**2)
        # p_head = np.arctan2(dy, dx)
        # alpha = wrapToPi(p_head - th)
        # gamma = wrapToPi(p_head - self.th_g)
        # V = self.k1 * phi * np.cos(alpha)
        # om = self.k2 * alpha + self.k1 * np.sinc(alpha/np.pi) * np.cos(alpha) * (alpha + self.k3 * gamma)
        # ########## Code ends here ##########
        # """change gamma and phi to rho and delta for publish"""
        # delta = gamma
        # rho = phi

        # # apply control limits
        # V = np.clip(V, -self.V_max, self.V_max)
        # om = np.clip(om, -self.om_max, self.om_max)

        # alpha_msg = Float64()
        # alpha_msg.data = alpha
        # self.pub_alpha.publish(alpha_msg)
        # delta_msg = Float64()
        # delta_msg.data = delta
        # self.pub_delta.publish(delta_msg)
        # rho_msg = Float64()
        # rho_msg.data = rho
        # self.pub_rho.publish(rho_msg)

        xt = x - self.x_g
        yt = y - self.y_g
        x = xt * np.cos(self.th_g) + yt * self.th_g * np.sinc(self.th_g/np.pi)
        y = yt * np.cos(self.th_g) - xt * self.th_g * np.sinc(self.th_g/np.pi)
        th = th - self.th_g

        rho = np.sqrt(x ** 2 + y ** 2)
        alpha = wrapToPi(np.arctan2(y, x) - th + np.pi)
        delta = wrapToPi(alpha + th)
        V = self.k1 * rho * np.cos(alpha)
        om = self.k2 * alpha + self.k1 * np.sinc(2 * alpha / np.pi) * (alpha + self.k3 * delta)
        ########## Code ends here ##########
        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)
        
        return V, om