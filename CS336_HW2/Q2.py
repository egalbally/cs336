import numpy as np
# from Q1 import Q1_solution #to locally run the code
from .Q1 import Q1_solution #for submission


class Q2_solution(Q1_solution): #Q2_solution inherits all attributes and functions from Q1_solution
  
  observation_dim = 3
    
  @staticmethod
  def observation(x):
    """ Implement Q2A. Observation function without noise.
    Input:
      x: (6,) numpy array representing the state.
    Output:
      obs: (3,) numpy array representing the observation (u,v,d).
    Note:
      we define disparity to be possitive.
    """
    # x_h' = k * x_pos
    x_pos = x[0:3]
    k = Q1_solution.k_camera # k_camera is an attribute of the parent class Q1_solution (self.k_camera would also work)
    x_pos_h = np.dot(k, x_pos)
    b = 0.2
    f = k[0,0]
    
    # observation
    u = x_pos_h[0]/x_pos_h[2]
    v = x_pos_h[1]/x_pos_h[2]
    d = f*b/x_pos_h[2]  # disparity = focus * baseline / depth
    obs = np.array([u,v,d])
   
    return obs

  @staticmethod
  def observation_state_jacobian(x):
    """ Implement Q2B. The jacobian of observation function w.r.t state.
    Input:
      x: (6,) numpy array, the state to take jacobian with.
    Output:
      H: (3,6) numpy array, the jacobian H.
    """
    f = 500
    b = 0.2
    xi = x[0]
    y = x[1]
    z = x[2]
    H = np.array([[f/z, 0,  -xi*f/(z*z), 0,0,0],
                  [0,   f/z, -y*f/(z*z), 0,0,0],
                  [0,   0,   -b*f/(z*z), 0,0,0]])
    return H

  @staticmethod
  def observation_noise_covariance():
    """ Implement Q2C here.
    Output:
      R: (3,3) numpy array, the covariance matrix for observation noise.
    """
    var_u = 5
    var_v = 5
    var_d = 5+5
    cov_ud = var_u 
    cov_vd = 0
    cov_uv = 0
    R = np.array([[var_u,  cov_uv, cov_ud],
                  [cov_uv, var_v,  cov_vd],
                  [cov_ud, cov_vd, var_d]])         
    return R


'''----------------- TEST CODE ------------------------------'''
'''----------------------------------------------------------'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from plot_helper import draw_2d, draw_3d

    np.random.seed(315)
    solution = Q2_solution()
    states, observations = solution.simulation()
    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(states[:,0], states[:,1], states[:,2], c=np.arange(states.shape[0]))
    plt.show()

    fig = plt.figure()
    plt.scatter(observations[:,0], observations[:,1], c=np.arange(states.shape[0]), s=4)
    plt.xlim([0,640])
    plt.ylim([0,480])
    plt.gca().invert_yaxis()
    plt.show()

    observations = np.load('./data/Q2D_measurement.npy')
    filtered_state_mean, filtered_state_sigma, predicted_observation_mean, predicted_observation_sigma = \
        solution.EKF(observations)
    # plotting
    true_states = np.load('./data/Q2D_state.npy')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(true_states[:,0], true_states[:,1], true_states[:,2], c='C0')
#     for mean, cov in zip(filtered_state_mean, filtered_state_sigma):
#         draw_3d(ax, cov[:3,:3], mean[:3])
    ax.view_init(elev=10., azim=45)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(observations[:,0], observations[:,1], s=4)
#     for mean, cov in zip(predicted_observation_mean, predicted_observation_sigma):
#         draw_2d(ax, cov[:2,:2], mean[:2])
    plt.xlim([0,640])
    plt.ylim([0,480])
    plt.gca().invert_yaxis()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(observations[:,0], observations[:,1]-observations[:,2], s=4)
#     for mean, cov in zip(predicted_observation_mean, predicted_observation_sigma):
        # TODO find out the mean and convariance for (u^R, v^R).
#         raise NotImplementedError()
#         draw_2d(ax, right_cov, right_mean)
    plt.xlim([0,640])
    plt.ylim([0,480])
    plt.gca().invert_yaxis()
    plt.show()




