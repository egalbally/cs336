import numpy as np

class Q1_solution(object):

  k_camera = np.array([[500,0,320],
                         [0,500,240],
                         [0,0,1]])
  observation_dim = 2

  @staticmethod
  def system_matrix():
    """ Implement the answer to Q1A here.
    Output:
      A: 6x6 numpy array for the system matrix.
    """
    t = 0.1
    A = np.array([[1,0,0,t,0,0],
                  [0,1,0,0,t,0],
                  [0,0,1,0,0,t],
                  [0,0,0,0.8,0,0],
                  [0,0,0,0,0.8,0],
                  [0,0,0,0,0,0.8]])
    return A

  @staticmethod
  def process_noise_covariance():
    """ Implement the covariance matrix Q for process noise.
    Output:
      Q: 6x6 numpy array for the covariance matrix.
    """
    # noise only in the vel part of the state which depends on a*t (not measurable)
    Q = np.zeros((6,6))
    Q[3,3] = Q[4,4] = Q[5,5] = 0.05 
    return Q

  @staticmethod
  def observation_noise_covariance():
    """ Implement the covariance matrix R for observation noise.
    Output:
      R: 2x2 numpy array for the covariance matrix.
    """
    R = np.array([[5, 0],
                  [0, 5]])
    return R

  @staticmethod
  def observation(state):
    """ Implement the function h, from state to noise-less observation. (Q1B)
    Input:
      state: (6,) numpy array representing state.
    Output:
      obs: (2,) numpy array representing observation.
    """
    state_p = state[0:3]
    obs_3d = np.dot(Q1_solution.k_camera,state_p)
    obs = np.array([obs_3d[0]/obs_3d[2], obs_3d[1]/obs_3d[2]])
    return obs

  def simulation(self, T=100):
    """ simulate with fixed start state for T timesteps.
    Input:
      T: an integer (=100).
    Output:
      states: (T,6) numpy array of states, including the given start state.
      observations: (T,2) numpy array of observations, Including the observation of start state.
    Note:
      We have set the random seed for you. Please only use np.random.multivariate_normal to sample noise.
      Keep in mind this function will be reused for Q2 by inheritance.
    """
    # Vars
    x0 = np.array([0.5, 0.0, 5.0, 0.0, 0.0, 0.0]) 
    A = self.system_matrix()
    #   -observations
    observation_size = self.observation_dim
    obs_mean = np.zeros((observation_size,))
    obs_cov = self.observation_noise_covariance() #(2x2)
    observations = np.zeros((T,observation_size)) 
    #   -states
    predState_mean = np.zeros((6,))
    predState_cov = self.process_noise_covariance() #(6x6)    
    states = np.zeros((T,6))
        
    # States and observations
    for i in range(T):
        if i == 0:
            states[i,:] = x0
            obs_noise = np.random.multivariate_normal(obs_mean, obs_cov)
            observations[i,:] = self.observation(x0) + obs_noise
        else:
            predState_noise = np.random.multivariate_normal(predState_mean, predState_cov)
            states[i,:] = np.dot(A,states[i-1,:]) + predState_noise
            obs_noise = np.random.multivariate_normal(obs_mean, obs_cov)
            observations[i,:] = self.observation(states[i,:]) + obs_noise
    
    return states, observations

  @staticmethod
  def observation_state_jacobian(x):
    """ Implement your answer for Q1D.
    Input:
      x: (6,) numpy array, the state we want to do jacobian at.
    Output:
      H: (2,6) numpy array, the jacobian of the observation model w.r.t state.
    """
    fx = 500
    fy = 500
    xi = x[0]
    y = x[1]
    z = x[2]
    H = np.array([[fx/z, 0, -xi*fx/(z*z), 0,0,0],
                  [0, fy/z, -y*fy/(z*z), 0,0,0]])
    return H

  def EKF(self, observations):
    """ Implement Extended Kalman filtering (Q1E)
    Input:
      observations: (N,2) numpy array, the sequence of observations. From T=1.
      mu_0: (6,) numpy array, the mean of state belief after T=0
      sigma_0: (6,6) numpy array, the covariance matrix for state belief after T=0.
    Output:
      state_mean: (N,6) numpy array, the filtered mean state at each time step. Not including the
                  starting state mu_0.
      state_sigma: (N,6,6) numpy array, the filtered state covariance at each time step. Not including
                  the starting state covarance matrix sigma_0.
      predicted_observation_mean: (N,2) numpy array, the mean of predicted observations. Start from T=1
      predicted_observation_sigma: (N,2,2) numpy array, the covariance matrix of predicted observations. Start from T=1
    Note:
      Keep in mind this function will be reused for Q2 by inheritance.  
    """    
    # Initial conditions
    mu_0 = np.array([0.5, 0.0, 5.0, 0.0, 0.0, 0.0])
    sigma_0 = np.eye(6)*0.01
    sigma_0[3:,3:] = 0.0
 
    # sizes of stuff 
    N = observations.shape[0]
    observation_size = self.observation_dim
    
    # Other filter matrices
    A = self.system_matrix()
    R = self.observation_noise_covariance()
    Q = self.process_noise_covariance()
    I = np.eye(6)
    predicted_observation_mean = np.zeros((N,observation_size)) # h(x_hat)
    predicted_observation_sigma = np.zeros((N,observation_size,observation_size)) # HPH_t + R = Kdenominator       
    state_sigma = np.zeros((N,6,6)) # sigma_pred
    state_mean = np.zeros((N,6)) # mu_pred
    
    # Extended Kalman Filter Iterations
    for t in range(N):
        #  - Prediction
        if t == 0:
            pred_state_mean = np.dot(A,mu_0) 
            pred_state_sigma = np.dot(np.dot(A,sigma_0), A.transpose()) + Q       
        else:
            pred_state_mean = np.dot(A,state_mean[t-1]) 
            pred_state_sigma = np.dot(np.dot(A,state_sigma[t-1]), A.transpose()) + Q
        
        #  - Update with new observation and Kalman gain
        H = self.observation_state_jacobian(pred_state_mean)
        knum = np.dot(pred_state_sigma,H.transpose())
        kdenom = np.dot(H,np.dot(pred_state_sigma,H.transpose())) + R
        kgain = np.dot(knum, np.linalg.inv(kdenom))
        
        predicted_observation_sigma[t] = kdenom
        predicted_observation_mean[t] = self.observation(pred_state_mean)
        
        state_mean[t] = pred_state_mean + np.dot(kgain, (observations[t] - predicted_observation_mean[t]))
        state_sigma[t] = np.dot((I - np.dot(kgain,H)), pred_state_sigma)
        
    return state_mean, state_sigma, predicted_observation_mean, predicted_observation_sigma


'''----------------- TEST CODE ------------------------------'''
'''----------------------------------------------------------'''

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from plot_helper import draw_2d, draw_3d
    # TODO: you also need to implement the draw_2d and draw_3d functions.

    np.random.seed(402)
    solution = Q1_solution()
    states, observations = solution.simulation()
    
    # ------ plotting
    
    #fig.1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(states[:,0], states[:,1], states[:,2], c=np.arange(states.shape[0]))
    fig.savefig("hw2_q1_1.png",dpi=600)
    plt.show()
    
    #fig.2
    fig = plt.figure()
    plt.scatter(observations[:,0], observations[:,1], c=np.arange(states.shape[0]), s=4)
    plt.xlim([0,640])
    plt.ylim([0,480])
    plt.gca().invert_yaxis()
    fig.savefig("hw2_q1_2.png",dpi=600)
    plt.show()

    observations = np.load('./data/Q1E_measurement.npy')
    filtered_state_mean, filtered_state_sigma, predicted_observation_mean, predicted_observation_sigma = \
        solution.EKF(observations)
    
    # ------- plotting
    true_states = np.load('./data/Q1E_state.npy')
    
    #fig.3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(true_states[:,0], true_states[:,1], true_states[:,2], c='C0')
#     for mean, cov in zip(filtered_state_mean, filtered_state_sigma):
#         draw_3d(ax, cov[:3,:3], mean[:3])
    ax.view_init(elev=10., azim=30)
    plt.show()
#     plt.save("hw2_q1_3.png",ax)

    
    #fig.4
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(observations[:,0], observations[:,1], s=4)
#     for mean, cov in zip(predicted_observation_mean, predicted_observation_sigma):
#         draw_2d(ax, cov, mean)
    plt.xlim([0,640])
    plt.ylim([0,480])
    plt.gca().invert_yaxis()
    plt.show()
#     plt.save("hw2_q1_4.png",ax)



