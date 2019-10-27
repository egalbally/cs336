import numpy as np
from Q1 import Q1_solution #to locally run the code
# from .Q1 import Q1_solution #for submission


class Q4_solution(Q1_solution): 
  
  observation_dim = 3

  @staticmethod
  def observation(x):
    """ Implement Q2A. Observation function without noise.
    Input:
      x: (6,) numpy array representing the state.
    Output:
      obs: (3,) numpy array representing the observation (x,y,z).
    Note:
      we define disparity to be possitive.
    """
    obs = x[0:3]
   
    return obs

  @staticmethod
  def observation_state_jacobian(x):
    """ Implement Q2B. The jacobian of observation function w.r.t state.
    Input:
      x: (6,) numpy array, the state to take jacobian with.
    Output:
      H: (3,6) numpy array, the jacobian H.
    """
    H = np.block([np.eye(3),np.zeros((3,3))])
    return H

  @staticmethod
  def observation_noise_covariance():
    """ Implement Q2C here.
    Output:
      R: (3,3) numpy array, the covariance matrix for observation noise.
    """
    R = np.diag([2,2,2])         
    return R


'''-----------------   Plots   ------------------------------'''
'''----------------------------------------------------------'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from plot_helper import draw_2d, draw_3d

    np.random.seed(315)
    solution = Q4_solution()
    q4flag = False #no outlier rejection
    
#     '''--------------------------------------------------------
#         4B) Estimated trajectory, detections and ground truth
#     -----------------------------------------------------------'''   
#     true_states = np.load('./data/Q4B_data/Q4B_positions_gt.npy')
#     observations = np.load('./data/Q4obsB.npy')
#     filtered_state_mean, filtered_state_sigma, predicted_observation_mean, predicted_observation_sigma = \
#         solution.EKF(observations,q4flag)   
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(true_states[:,0], true_states[:,1], true_states[:,2], label='ground truth') 
#     ax.scatter(filtered_state_mean[:,0], filtered_state_mean[:,1], filtered_state_mean[:,2], label='estimated trajectory') 
#     ax.scatter(observations[:,0], observations[:,1], observations[:,2], label='detections') 
#     ax.view_init(elev=10., azim=45)
#     ax.set_xlabel('$X$')
#     ax.set_ylabel('$Y$')
#     ax.set_zlabel('$Z$')
#     leg = ax.legend();
#     fig.savefig("hw2_q4B.png",dpi=600)
#     plt.show()

#     #----------- Not needed
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(true_states[:,0], true_states[:,1], true_states[:,2], label='ground truth') 
#     ax.scatter(filtered_state_mean[:,0], filtered_state_mean[:,1], filtered_state_mean[:,2], label='estimated trajectory') 
#     ax.view_init(elev=10., azim=45)
#     ax.set_xlabel('$X$')
#     ax.set_ylabel('$Y$')
#     ax.set_zlabel('$Z$')
#     leg = ax.legend();
#     fig.savefig("hw2_q4B_noDetections.png",dpi=600)
#     plt.show()
    
#     '''----------------------------------------------------------
#         4C) Occlusions: 
#             - Estimated trajectory and ground truth 
#             - Estimated 3D trajectory in 2D (x vs. z) (y vs. z)
#     -------------------------------------------------------------'''
#     true_states = np.load('./data/Q4D_data/Q4D_positions_gt.npy')
#     observations = np.load('./data/Q4obsD.npy')
#     filtered_state_mean, filtered_state_sigma, predicted_observation_mean, predicted_observation_sigma = \
#         solution.EKF(observations, q4flag)   
                          
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(true_states[:,0], true_states[:,1], true_states[:,2], label='ground truth') #ground truth
#     ax.scatter(filtered_state_mean[:,0], filtered_state_mean[:,1], filtered_state_mean[:,2], label='estimated trajectory') #estimated
#     ax.view_init(elev=10., azim=45)
#     ax.set_xlabel('$X$')
#     ax.set_ylabel('$Y$')
#     ax.set_zlabel('$Z$')
#     leg = ax.legend();
#     fig.savefig("hw2_q4C.png",dpi=600)
#     plt.show()
    
#     #----------- Not needed
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(true_states[:,0], true_states[:,1], true_states[:,2], label='ground truth') #ground truth
#     ax.scatter(filtered_state_mean[:,0], filtered_state_mean[:,1], filtered_state_mean[:,2], label='estimated trajectory') #estimated
#     ax.scatter(observations[:,0], observations[:,1], observations[:,2], label='detections') 
#     ax.view_init(elev=10., azim=45)
#     ax.set_xlabel('$X$')
#     ax.set_ylabel('$Y$')
#     ax.set_zlabel('$Z$')
#     leg = ax.legend();
#     fig.savefig("hw2_q4C_withDetections.png",dpi=600)
#     plt.show()
#     #-----------
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(true_states[:,0], true_states[:,2], label='ground truth') #ground truth
#     ax.scatter(filtered_state_mean[:,0], filtered_state_mean[:,2], label='estimated trajectory') #estimated
#     ax.set_xlabel('$X$')
#     ax.set_ylabel('$Z$')
#     leg = ax.legend();
#     fig.savefig("hw2_q4C_xz.png",dpi=600)
#     plt.show()
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(true_states[:,1], true_states[:,2], label='ground truth') #ground truth
#     ax.scatter(filtered_state_mean[:,1], filtered_state_mean[:,2], label='estimated trajectory') #estimated
#     ax.set_xlabel('$Y$')
#     ax.set_ylabel('$Z$')
#     leg = ax.legend();
#     fig.savefig("hw2_q4C_yz.png",dpi=600)
#     plt.show()
    
    '''--------------------------------------------------------------------------
        4D) Outlier rejection:
            - 2D projection of the observation trajectory with labelled outliers
            - ground truth projection 
    -----------------------------------------------------------------------------'''
    true_states = np.load('./data/Q4D_data/Q4D_positions_gt.npy')
    observations = np.load('./data/Q4obsD.npy')
    q4flag = True # outlier rejection
    filtered_state_mean, filtered_state_sigma, predicted_observation_mean, predicted_observation_sigma, outlier_indexes = \
        solution.EKF(observations, q4flag)   
    
    """---------  3D   ----------------"""                         
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')   
    ax.scatter(true_states[:,0], true_states[:,1], true_states[:,2], s=5, label='ground truth') #ground truth
    ax.scatter(filtered_state_mean[:,0], filtered_state_mean[:,1], filtered_state_mean[:,2], s=5, label='estimated trajectory') #estimated
    ax.scatter(observations[:,0], observations[:,1], observations[:,2], s=5, label='detections') 
    ax.scatter(observations[outlier_indexes,0], observations[outlier_indexes,1], observations[outlier_indexes,2], s=20, label='outliers')     
    ax.view_init(elev=10., azim=45)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    leg = ax.legend();
    fig.savefig("hw2_q4D_182out.png",dpi=600)
#     fig.savefig("hw2_q4D_152out.png",dpi=600)
    plt.show()

    """---------  2D   ----------------"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    N = true_states.shape[0]
    true_state_cam = np.zeros((N,2))
    filtered_state_mean_cam = np.zeros((N,2))
    observations_cam =  np.zeros((N,2))
    outliers_cam =  np.zeros((N,2))   
    sol = Q1_solution()
    
    for i in range(N):
        true_state_cam[i] = Q1_solution.observation(true_states[i])
        filtered_state_mean_cam[i] = Q1_solution.observation(filtered_state_mean[i])
        observations_cam[i] = Q1_solution.observation(observations[i])
        
    ax.scatter(true_state_cam[:,0], true_state_cam[:,1], label='ground truth in cam') 
    ax.scatter(filtered_state_mean_cam[:,0], filtered_state_mean_cam[:,1], label='estimated trajectory in cam') 
    ax.scatter(observations_cam[:,0], observations_cam[:,1], label='detections in cam') 
    ax.scatter(observations_cam[outlier_indexes,0], observations_cam[outlier_indexes,1], label='outliers in cam') 
    ax.set_xlabel('$u$')
    ax.set_ylabel('$v$')
    leg = ax.legend();
#     fig.savefig("hw2_q4D_xz_152.png",dpi=600)
    fig.savefig("hw2_q4D_cam.png",dpi=600)
    plt.show()
    
#     
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(true_states[:,1], true_states[:,2], label='ground truth') #ground truth
#     ax.scatter(filtered_state_mean[:,1], filtered_state_mean[:,2], label='estimated trajectory') #estimated
#     ax.scatter(observations[:,1], observations[:,2], label='detections') 
#     ax.scatter( observations[outlier_indexes,1], observations[outlier_indexes,2], label='outliers') 
#     ax.set_xlabel('$Y$')
#     ax.set_ylabel('$Z$')
#     leg = ax.legend();
# #     fig.savefig("hw2_q4D_yz_152.png",dpi=600)
#     fig.savefig("hw2_q4D_yz_182.png",dpi=600)
#     plt.show()


