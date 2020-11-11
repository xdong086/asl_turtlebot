import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    t = np.zeros(len(path))
    t[1:] = np.cumsum(np.array([np.sqrt((path[i][0] - path[i-1][0]) ** 2 + (path[i][1] - path[i-1][1]) ** 2) / V_des  for i in range(1, len(path))]))
    spl_x = scipy.interpolate.splrep(t, [p[0] for p in path], s=alpha)
    spl_y = scipy.interpolate.splrep(t, [p[1] for p in path], s=alpha)
    t_smoothed = np.arange(0.0, t[-1], dt)
    traj_smoothed_x = scipy.interpolate.splev(t_smoothed, spl_x)
    traj_smoothed_y = scipy.interpolate.splev(t_smoothed, spl_y)
    traj_smoothed_xd = scipy.interpolate.splev(t_smoothed, spl_x, der=1)
    traj_smoothed_yd = scipy.interpolate.splev(t_smoothed, spl_y, der=1)
    traj_smoothed_xdd = scipy.interpolate.splev(t_smoothed, spl_x, der=2)
    traj_smoothed_ydd = scipy.interpolate.splev(t_smoothed, spl_y, der=2)
    traj_smoothed = np.zeros((len(t_smoothed), 7))
    traj_smoothed[:, 0] = traj_smoothed_x
    traj_smoothed[:, 1] = traj_smoothed_y
    traj_smoothed[:, 2] = np.array([np.arctan2(yd, xd) for xd, yd in zip(traj_smoothed_xd, traj_smoothed_yd)])
    traj_smoothed[:, 3] = traj_smoothed_xd
    traj_smoothed[:, 4] = traj_smoothed_yd
    traj_smoothed[:, 5] = traj_smoothed_xdd
    traj_smoothed[:, 6] = traj_smoothed_ydd
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed


