#!/usr/bin/env python3
import numpy as np
import scipy.linalg

# vars
Q = np.diag([0.01,0.001,0.001,0.0,0.005,0.0025])
R = np.diag([0.1,0.1,0.1,0.05,0.05,0.05])
t = 0.1                 

def twist_to_transform(twist, time): #from hw1
    vx,vy,vz,wx,wy,wz = twist
    w_skew_symm = np.array([[0,   -wz,   wy ],
                            [wz,   0,   -wx ],
                            [-wy,  wx,   0  ]])
    v = np.array([[vx],[vy],[vz]])
    twist_matrix = np.block([[w_skew_symm, v],
                            [0,0,0,0]])
    transform = scipy.linalg.expm(twist_matrix*time)
    return transform

def simulate(commands):
    """ Code for question 3a.
    Input:
      commands: (N,6) numpy arrays, robot commands
    Output:
      poses : (N+1, 4,4) numpy arrays, poses of the robot end effector. Include initial pose.
      twists: (N+1, 6) numpy arrays, twists of the robot end effector. Include initial twist.
    """
    pose_0 = np.eye(4)
    pose_0[:3,3] = np.array([1.0, 2.0, 5.0])
    twist_0 = np.zeros(6)
    
    N = commands.shape[0]
    poses = np.zeros((N+1, 4,4))
    twists = np.zeros((N+1, 6))
        
    for i in range(N):
        if i == 0:
            poses[i] = pose_0
            twists[i] = twist_0
        else:
            twist_mat = twist_to_transform(twists[i-1], t)
            poses[i] = np.dot(twist_mat, poses[i-1])
            twist_noise = np.random.multivariate_normal(np.zeros((6,)), Q)
            twists[i] = twists[i-1] + t*commands[i-1] + twist_noise
            
    return poses, twists


def particle_filter(observations, commands):
    """ Code for question 3b.
    Input:
      observations: (N,2) numpy arrays, observations from measurements. Starting from T=1
      commands: (N,6) numpy arrays, robot commands
      pose_0: (4,4) numpy arrays, starting pose
    Output:
      max_likelihood_pose : (N, 4,4) numpy arrays, estimated pose from the filter
      max_prob: (N) numpy arrays, probability
    """
    # init
    pose_0 = np.eye(4)
    pose_0[:3,3] = np.array([1.0, 2.0, 5.0])
    twist_0 = np.zeros(6)
    
    # vars
    num_part = 50
    weights = np.ones(num_part)/num_part
    num_iter = observations.shape[0]
    
    for k in range(num_iter):
        for i in range(num_part):
            twist_mat = twist_to_transform(twists[i-1], k)
            poses[i] = np.dot(twist_mat, poses[i-1])
            twist_noise = np.random.multivariate_normal(np.zeros((6,)), Q)
            twists[i] = twists[i-1] + t*commands[i-1] + twist_noise
        # resample
        # redistribute particles according to normal distribution
    
    return max_prob_pose, max_prob


'''----------------- TEST CODE ------------------------------'''
'''----------------------------------------------------------'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from plot_helper import draw_2d, draw_3d

    np.random.seed(123)
    commands = np.load('./data/Q3_data/q3_commands.npy')
    poses, twists = simulate(commands)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(poses[:,0,3], poses[:,1,3], poses[:,2,3],
              poses[:,0,0], poses[:,1,0], poses[:,2,0],
              color='b', length=0.3, arrow_length_ratio=0.05)
    ax.quiver(poses[:,0,3], poses[:,1,3], poses[:,2,3],
              poses[:,0,1], poses[:,1,1], poses[:,2,1],
              color='g', length=0.3, arrow_length_ratio=0.05)
    ax.quiver(poses[:,0,3], poses[:,1,3], poses[:,2,3],
              poses[:,0,2], poses[:,1,2], poses[:,2,2],
              color='r', length=0.3, arrow_length_ratio=0.05)
    ax.set_xlim([-5.0, 4.0])
    ax.set_ylim([-5.0, 4.0])
    ax.set_zlim([0.0, 5.0])
    fig.savefig("hw2_q3_sim.png",dpi=600)
    plt.show()

    observations = np.load('./data/Q3_data/q3_measurements.npy')
    max_prob_pose, max_prob = particle_filter(observations, commands)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(max_prob_pose[:,0,3], max_prob_pose[:,1,3], max_prob_pose[:,2,3],
              max_prob_pose[:,0,0], max_prob_pose[:,1,0], max_prob_pose[:,2,0],
              color='b', length=0.3, arrow_length_ratio=0.05)
    ax.quiver(max_prob_pose[:,0,3], max_prob_pose[:,1,3], max_prob_pose[:,2,3],
              max_prob_pose[:,0,1], max_prob_pose[:,1,1], max_prob_pose[:,2,1],
              color='g', length=0.3, arrow_length_ratio=0.05)
    ax.quiver(max_prob_pose[:,0,3], max_prob_pose[:,1,3], max_prob_pose[:,2,3],
              max_prob_pose[:,0,2], max_prob_pose[:,1,2], max_prob_pose[:,2,2],
              color='r', length=0.3, arrow_length_ratio=0.05)
    ax.set_xlim([-1.0, 2.0])
    ax.set_ylim([-1.0, 2.0])
    ax.set_zlim([4.0, 7.0])
#     fig.savefig("hw2_q3_filter.png",dpi=600)
    plt.show()
