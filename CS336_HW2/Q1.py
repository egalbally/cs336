import numpy as np

class Q1_solution(object):

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
                  [0,0,0,1,0,0],
                  [0,0,0,0,1,0],
                  [0,0,0,0,0,1]])
    print("A(6,6)", A.shape())
    return A

  @staticmethod
  def process_noise_covariance():
    """ Implement the covariance matrix Q for process noise.
    Output:
      Q: 6x6 numpy array for the covariance matrix.
    """
    # noise only in the vel part of the state which depends on a*t (not measurable)
    Q = np.zeros((6,6))
    Q[4,4] = Q[5,5] = Q[6,6] = 0.5 
    print("Q(6,6)", Q.size())
    return Q

  @staticmethod
  def observation_noise_covariance():
    """ Implement the covariance matrix R for observation noise.
    Output:
      R: 2x2 numpy array for the covariance matrix.
    """
    R = np.array([[5, 0],
                  [0, 5]])
    print("R(2,2)", R.size())
    return R

  @staticmethod
  def observation(state):
    """ Implement the function h, from state to noise-less observation. (Q1B)
    Input:
      state: (6,) numpy array representing state.
    Output:
      obs: (2,) numpy array representing observation.
    """
    state_p = np.reshape(state[0:4], (3,1))
    k_camera = np.array([[500,0,320],
                         [0,500,240],
                         [0,0,1]])
    obs_3d = np.dot(k_camera,state_p)
    obs = (obs_3d[0]/obs_3d[2], obs_3d[1]/obs_3d[2])
    print("obs(2,)", obs.size())
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
    #     the order to generate random noise should be: observation noise for T=0, process noise for T=1, observation noise for T=1, ...
    #     Q1C: Please use the 6D process noise covariance when generating process noise. Using the 3x3 submatrix will not match our implementation.
    
    # Vars
    x_0 = np.array([0.5, 0.0, 5.0, 0.0, 0.0, 0.0])  
    meas_mean = [0, 0]
    meas_cov = observation_noise_covariance() # (2x2)
    predState_mean = np.zeros((1,6))
    predState_cov = process_noise_covariance() #(6x6)    
    states = np.zeros((T,6))
    observations = np.zeros((T,2)) 
    
    # States and observations
    for i in range(T):
        if i == 0:
            states[i,:] = x_0
            meas_noise = np.random.multivariate_normal(meas_mean, meas_cov)
            observations[i,:] = observation(x0) + meas_noise
        else:
            predState_noise = np.random.multivariate_normal(predState_mean, predState_cov)
            meas_noise = np.random.multivariate_normal(meas_mean, meas_cov)
            states[i,:] = states[i-1,:] + predState_noise
            observations[i,:] = observation[i-1] + meas_noise
    
    # Plot
    statePlot = plt.axes(projection='3d')
    statePlot.plot3D(states[:,0], states[:,1], states[:,2], 'blue')
    plt.show()
    
#     fig, predictions = plt.subplots(1, 2)
#     predictions[0].plot3D(x_t0, y_t0, color = color_t0/255)
#     predictions[0].set_title('predicted position')
#     predictions[1].scatter(x_t2, y_t2, color = color_t2/255)
#     predictions[1].set_title('observations')
#     plt.show()
    
#     print("states: (100,6)", states.shape())
#     print("observations: (100,2)", observations.shape())
    
    return states, observations

  @staticmethod
  def observation_state_jacobian(x):
    """ Implement your answer for Q1D.
    Input:
      x: (6,) numpy array, the state we want to do jacobian at.
    Output:
      H: (2,6) numpy array, the jacobian of the observation model w.r.t state.
    """
    raise NotImplementedError
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
    
    #     Q1E: when plotting ellipse and ellipsoids, scale them to be confidence interval 95%
    # As a reference, the code below generates the surface mesh numpy array for a sphere with radius 1 and centered at the origin. You should scale,rotate,and translate this sphere to be the error ellipsoid.

        # u = np.linspace(0.0, 2.0 * np.pi, 10)
        # v = np.linspace(0.0, np.pi, 10)
        # x = np.outer(np.cos(u), np.sin(v))
        # y = np.outer(np.sin(u), np.sin(v))
        # z = np.outer(np.ones_like(u), np.cos(v))</code>
        #     mu_0 = np.array([0.5, 0.0, 5.0, 0.0, 0.0, 0.0])
        #     sigma_0 = np.eye(6)*0.01
        #     sigma_0[3:,3:] = 0.0
    

    mu_0 = np.array([0.5, 0.0, 5.0, 0.0, 0.0, 0.0])
    sigma_0 = np.eye(6)*0.01
    sigma_0[3:,3:] = 0.0

    
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

    observations = np.load('./data/Q1E_measurement.npy')
    filtered_state_mean, filtered_state_sigma, predicted_observation_mean, predicted_observation_sigma = \
        solution.EKF(observations)
    # plotting
    true_states = np.load('./data/Q1E_state.npy')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(true_states[:,0], true_states[:,1], true_states[:,2], c='C0')
    for mean, cov in zip(filtered_state_mean, filtered_state_sigma):
        draw_3d(ax, cov[:3,:3], mean[:3])
    ax.view_init(elev=10., azim=30)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(observations[:,0], observations[:,1], s=4)
    for mean, cov in zip(predicted_observation_mean, predicted_observation_sigma):
        draw_2d(ax, cov, mean)
    plt.xlim([0,640])
    plt.ylim([0,480])
    plt.gca().invert_yaxis()
    plt.show()



