#!/usr/bin/env python3
import numpy as np
import scipy.linalg
import scipy.stats

# vars
Q = np.diag([0.01,0.001,0.001,0.0,0.005,0.0025])
R = np.diag([0.1,0.1,0.1,0.05,0.05,0.05])
delta_t = 0.1                 

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

def transform_to_twist(T_pose):
    """ From hw1
    Input:
      T_pose: (4,4) numpy array. The result from Q1_H.
    Output:
      twist: a np array (vx,vy,vz,wx,wy,wz).
    """  
    # 1) T = e^[V] -> [V] = Ln(T): 
    twist_skew = np.real(scipy.linalg.logm(T_pose))
    
    # 2) From skew symm. to vector:
    twist = np.array([twist_skew[0,3], twist_skew[1,3],twist_skew[2,3],
             twist_skew[2,1], twist_skew[0,2], twist_skew[1,0]])
    
    return twist

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
        
    for t in range(N): #we have 1 particle and propagate it in time
        if t == 0:
            poses[t] = pose_0
            twists[t] = twist_0
        else:
            twist_mat = twist_to_transform(twists[t-1], delta_t)
            poses[t] = np.dot(twist_mat, poses[t-1])
            twist_noise = np.random.multivariate_normal(np.zeros((6,)), Q)
            twists[t] = twists[t-1] + delta_t*commands[t-1] + twist_noise
            
    return poses, twists


def particle_filter(observations, commands):
    """ Code for question 3b.
    Input:
      observations: (N,2) numpy arrays, observations from measurements. Starting from T=1
      commands: (N,6) numpy arrays, robot commands
      pose_0: (4,4) numpy arrays, starting pose
    Output:
      max_prob_pose : (N, 4,4) numpy arrays, estimated pose from the filter
      max_prob: (N) numpy arrays, probability
    """
    # init
    pose_0 = np.eye(4)
    pose_0[:3,3] = np.array([1.0, 2.0, 5.0])
    twist_0 = np.zeros(6)
    
    # vars
    num_part = 100
    num_iter = observations.shape[0] # number of timesteps we will forward propagate through
    weights = np.ones(num_part)/num_part # normalized to 1
    max_prob_pose = np.zeros((num_iter, 4,4))
    max_prob = np.zeros(num_iter)
    poses = np.zeros((num_part, 4,4)) #contains all particles for one time step and gets overwritten at t+1
    twists = np.zeros((num_part, 6))
    
    num_total_de_quesitos_regulares = 200
    pie_in_vector_form = np.zeros(num_total_de_quesitos_regulares,dtype=int)
    
    for t in range(num_iter):
        for i in range(num_part):
            '''---------------------------------
                  Update particle values @t
               ---------------------------------'''          
            # Populate the initial values (t=0) for all particles
            if t == 0: 
                poses[i] = pose_0 # we know (100% sure) that the initial (t=0) pose for all particles is pose_0
                twists[i] = twist_0
            # Update all particle values for the current time step:
            # each particle has associated to it a set of poses for t=0,...t=100
            else: 
                twist_mat = twist_to_transform(twists[i], delta_t)
                poses[i] = np.dot(twist_mat, poses[i])
                twist_noise = np.random.multivariate_normal(np.zeros((6,)), Q)
                twists[i] = twists[i] + delta_t*commands[t-1] + twist_noise
                         
            '''---------------------------------
                  Calculate particle weights @t
               ---------------------------------''' 
            # (1) Finding the difference in pose: T_obs*T_i^(-1)
            delta_pose_4x4 = observations[t].dot(np.linalg.inv(poses[i]))
            # (2) Transform  to 6D delta pose vector
            delta_pose_twist = transform_to_twist(delta_pose_4x4)
            # (3) Give big weights to the smallest delta pose twist vectors:
            #     the weight values come from sampling a 6 dimensional gaussian distribution
            weights[i] = weights[i]* scipy.stats.multivariate_normal.pdf(delta_pose_twist,cov=R)

            # Save particle with biggest weight for output
            if(weights[i] > max_prob[t]):
                max_prob[t] = weights[i]
                max_prob_pose[t] = poses[i] 

        # Normalize all weights at once:
        weights = weights/np.sum(weights)
        
        '''-----------------------------------------------------------
              Resample particle values @t+1  --> La ruleta de la muerte 
           -----------------------------------------------------------'''
        idx = 0
        for i in range(num_part):
            # weight_i is to 1 like num_de_quesitos_por_particle is to num_total_de_quesitos_regulares
            num_de_quesitos_particle = int(np.floor(weights[i]*num_total_de_quesitos_regulares))
            # transformar los quesitos de la ruleta en un vector
            pie_in_vector_form[idx:idx+num_de_quesitos_particle] = i 
            idx = idx + num_de_quesitos_particle
        for i in range(num_part):
            # generar numeros aleatorios and resample
            numero_magico = np.random.randint(0,num_total_de_quesitos_regulares)
            idx_particula_elegida = pie_in_vector_form[numero_magico]
            poses[i] = poses[idx_particula_elegida]
            weights[i] = weights[idx_particula_elegida]
            twists[i] = twists[idx_particula_elegida]

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
    
    #PLOT 1
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
    
    # PLOT 2
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
    ax.set_xlim([-5.0, 4.0])
    ax.set_ylim([-5.0, 4.0])
    ax.set_zlim([0.0, 5.0])
    fig.savefig("hw2_q3_filter.png",dpi=600)
    plt.show()
