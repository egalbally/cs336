import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pdb
from math import *

def quaternion_to_matrix(q):
    w,i,j,k = q
    return np.array([[1.0-2*j*j-2*k*k, 2*i*j-2*w*k, 2*i*k+2*w*j],
                     [2*i*j+2*w*k, 1.0-2*i*i-2*k*k, 2*j*k-2*w*i],
                     [2*i*k-2*w*j, 2*j*k+2*w*i, 1.0-2*i*i-2*j*j]])

def Q1_A(t_RF_WF, q_RF_WF, t_CF_RF, u_CF_RF, theta_CF_RF):
    """ Code for question 1a.
    Inputs:
      t_RF_WF: numpy array with shape (3,).
      q_RF_WF: tuple of (w,x,y,z) representing quaternion.
      t_CF_RF: numpy array with shape (3,).
      u_CF_RF: tuple of (x,y,z) representing the rotation axis.
      theta_CF_RF: float representing the rotation angle.
    Output:
      transform_RF_WF_t0: numpy array with shape (4,4) for transformation between RF and WF.
      transform_CF_RF_t0: numpy array with shape (4,4) for transformation between CF and RF.
      transform: numpy array with shape (4,4) for the transformation between CF and WF.
    """
    transform_RF_WF_t0 = np.block([[quaternion_to_matrix(q_RF_WF), np.reshape(t_RF_WF, (3,1))],
                                   [0,0,0,1]])
    
    q_CF_RF = (cos(theta_CF_RF/2) , u_CF_RF[0]*sin(theta_CF_RF/2), u_CF_RF[1]*sin(theta_CF_RF/2), u_CF_RF[2]*sin(theta_CF_RF/2))
    transform_CF_RF_t0 = np.block([[quaternion_to_matrix(q_CF_RF), np.reshape(t_CF_RF, (3,1))],
                                   [0,0,0,1]])
    
    transform =  np.dot(transform_RF_WF_t0, transform_CF_RF_t0) 
    return transform_RF_WF_t0, transform_CF_RF_t0, transform


def twist_to_transform(twist, time):
    vx,vy,vz,wx,wy,wz = twist
    w_skew_symm = np.array([[0,   -wz,   wy ],
                            [wz,   0,   -wx ],
                            [-wy,  wx,   0  ]])
    v = np.array([[vx],[vy],[vz]])
    twist_matrix = np.block([[w_skew_symm, v],
                            [0,0,0,0]])
    transform = scipy.linalg.expm(twist_matrix*time)
    return transform


def Q1_B(twist, time):
    """ Code for question 1b.
    Input:
      twist: tuple of (vx,vy,vz,wx,wy,wz) representing the twist in eq.7.
      time: a float representing the time duration (2.0s).
    Output:
      transform: numpy array with shape (4,4) for the required transformation.
    """
    transform = twist_to_transform(twist, time)
    return transform


def Q1_C(twist, time):
    """ Code for question 1c.
    Input:
      twist: tuple of (vx,vy,vz,wx,wy,wz) representing the twist in eq.8.
      time: a float representing the time duration (2.0s).
    Output:
      transform: numpy array with shape (4,4) for the required transformation.
    """
    transform = twist_to_transform(twist, time)
    return transform



def Q1_D(transform_RF_WF_0, transform_CF_RF_0, delta_transform_RF_WF, delta_transform_CF_RF):
    """ Code for question 1d.
    Input:
      transform_RF_WF_0: numpy array with shape (4,4). Output from Q1_A.
      transform_CF_RF_0: numpy array with shape (4,4). Output from Q1_A.
      delta_transform_RF_WF: numpy array with shape (4,4). Output from Q1_B.
      delta_transform_CF_RF: numpy array with shape (4,4). Output from Q1_C.
    Output:
      transform: numpy array with shape (4,4) for the required transformation.
    """
    transf_p0c_to_p2r = np.dot(delta_transform_CF_RF, transform_CF_RF_0)
    transf_p2r_to_p2wprime = np.dot(transform_RF_WF_0, transf_p0c_to_p2r)
    transform = np.dot(delta_transform_RF_WF, transf_p2r_to_p2wprime)
    return transform


def Q1_E(transform_CF_WF_t0, pc):
    """ Code for question 1e.
    Input:
      transform_CF_WF_0: (4,4) numpy array. The output from Q1_A.
      pc: (N,6) numpy array, loaded from pointcloud1.npy.
    Output:
      pc_position: (N,3) numpy array representing the point cloud in camera frame.
      pc_color: (N,3) numpy array representing the corresponding RGB values.
    """   
    pc_transpose = np.transpose(pc) # 6xN array
    rowOfOnes = np.transpose(np.ones(pc_transpose.shape[1]))
    pc_position_in_WF_h= np.block([[pc_transpose[0:3,:]],
                                    [rowOfOnes]]) # 4xN
    pc_position_in_CF_h = np.dot(np.linalg.inv(transform_CF_WF_t0), pc_position_in_WF_h) # (4x4)x(4xN) = 4xN
    pc_position = np.transpose(pc_position_in_CF_h[0:3,:]) # Nx3 array
    pc_color = pc[:, 3:] # Nx3 array
    #     print(pc_position[9,:])
    return pc_position, pc_color


def Q1_F(pc_position, pc_color, intrinsic):
    """ Code for question 1f.
    Input:
      pc_position: (N,3) numpy array. The result from Q1_E.
      pc_color: (N,3) numpy array. The result from Q1_E.
      intrinsic: (3,3) numpy array. The intrinsic matrix.
    Output:
      pc_position: (N,2) numpy array representing the pixel coordinates of points.
      pc_color: (N,3) numpy array representing the corresponding RGB values.
    """
    # homogeneous to euclidian
    pc_pos2d_h = np.transpose(np.dot(intrinsic,np.transpose(pc_position))) #Nx3
    pc_pos2d = np.transpose(np.array([pc_pos2d_h[:,0]/pc_pos2d_h[:,2], pc_pos2d_h[:,1]/pc_pos2d_h[:,2]])) #Nx2
    
    # variables
    j = 0;
    pc_position_crop = np.zeros(pc_pos2d.shape)
    pc_color_crop = np.zeros(pc_color.shape)
        
    # find points that meet the criteria
    for i in range(pc_pos2d.shape[0]):
         if (pc_pos2d[i,0]<640 and pc_pos2d[i,1]<480 and pc_pos2d[i,0] > 0 and pc_pos2d[i,1] > 0):
            pc_position_crop[j,:] = pc_pos2d[i,:]
            pc_color_crop[j,:] = pc_color[i,:]
            j = j+1

    # output
    pc_position = pc_position_crop[0:j, :] #jx2
    pc_color = pc_color_crop[0:j,:] #jx3
    return pc_position, pc_color


def positionInCamera_tx(transform_CF_WF_tx, intrinsic):
    pc1 = np.load('pointcloud1.npy')
    pc_position, pc_color = Q1_E(transform_CF_WF_tx, pc1)
    pc_position, pc_color = Q1_F(pc_position, pc_color, intrinsic)
    return pc_position, pc_color


def Q1_G(transform_CF_WF_t0, transform_CF_WF_t2, intrinsic):
    """ Code for question 1g.
    Input:
      transform_CF_WF_t0: (4,4) numpy array. The result from Q1_A.
      transform_CF_WF_t2: (4,4) numpy array. The result from Q1_D.
      intrinsic: (3,3) numpy array. The intrinsic matrix.
    Output:
      None. Include plot figures in separate PDF.
    """
    #data
    pc_position_t0, pc_color_t0 = positionInCamera_tx(transform_CF_WF_t0, intrinsic)
    pc_position_t2, pc_color_t2 = positionInCamera_tx(transform_CF_WF_t2, intrinsic)
    #     print(pc_position_t0[9])
    x_t0 = pc_position_t0[:,0]
    y_t0 = pc_position_t0[:,1]
    x_t2 = pc_position_t2[:,0]
    y_t2 = pc_position_t2[:,1]
    color_t0 = pc_color_t0
    color_t2 = pc_color_t2
    
    #plot
    fig, rgb_pixels = plt.subplots(1, 2)
    rgb_pixels[0].scatter(x_t0, y_t0, color = color_t0/255)
    rgb_pixels[0].set_title('time = 0')
    rgb_pixels[1].scatter(x_t2, y_t2, color = color_t2/255)
    rgb_pixels[1].set_title('time = 2')
    plt.show()
    

def find_transformation(pt_1A, pt_1B, pt_1C, pt_2A, pt_2B, pt_2C):
    """ Code for finding the 6D transformation from 3 point pairs.
    Input:
      pt_??: (3,1) numpy array of point positions in world frame.
      pt_1A corresponds to pt_2A, etc.
      The three points may be colinear. In which case you should return None.
    Output:
      transform: (4,4) numpy array representing the required transformation
        from pt_1? to pt_2?.
    """
    #vars
    pt_1A = np.reshape(pt_1A, (3,1))
    pt_1B = np.reshape(pt_1B, (3,1))
    pt_1C = np.reshape(pt_1C, (3,1))
    pt_2A = np.reshape(pt_2A, (3,1))
    pt_2B = np.reshape(pt_2B, (3,1))
    pt_2C = np.reshape(pt_2C, (3,1))
    centroid1 = centroid2 = np.zeros(pt_1A.shape)
    
    # centroids
    centroid1 = (pt_1A+pt_1B+pt_1C)/ 3
    centroid2 = (pt_2A+pt_2B+pt_2C)/ 3
    
    # H (3x3)
    H = (np.dot((pt_1A - centroid1),np.transpose(pt_2A - centroid2)) + 
         np.dot((pt_1B - centroid1),np.transpose(pt_2B - centroid2)) + 
         np.dot((pt_1C - centroid1),np.transpose(pt_2C - centroid2)))
    
    # Rotation, R (3x3)
    u, s, vh = np.linalg.svd(H) # u(9x9), s(6x1), vh(6x6)  
    v = vh.transpose()
    R = np.dot(v, np.transpose(u)) 
    # svd fix
    if np.linalg.det(R)<0:
        v[:,2] = -v[:,2]
        R = np.dot(v, np.transpose(u)) 
        
    # Translation, t (3x1)
    t = np.dot(-R, centroid1) + centroid2
    
    # Transformation, T
    transform = np.block([[R,t],
                         [0,0,0,1]])
    
    return transform


def Q1_H(idx1,idx2,idx3):
    """ Code for question 1h.
    Input:
      idx1, idx2, idx3: integers for the indices of corresponding point ids.
    Output:
      transform: (4,4) array representing the required transformation.
    """
    pc1 = np.load('pointcloud1.npy')
    pc2 = np.load('pointcloud2.npy')
    id1 = np.load('pointcloud1_ids.npy')
    id2 = np.load('pointcloud2_ids.npy')
    reorder_pc1 = np.zeros((pc1.shape[0], 3))
    for id,pt in zip(id1,pc1):
        reorder_pc1[id]=pt[:3]
    reorder_pc2 = np.zeros((pc2.shape[0], 3))
    for id,pt in zip(id2,pc2):
        reorder_pc2[id]=pt[:3]
    return find_transformation(reorder_pc1[idx1], reorder_pc1[idx2], reorder_pc1[idx3],
                               reorder_pc2[idx1], reorder_pc2[idx2], reorder_pc2[idx3])


def find_transformation_for_variable_size_pc(pc1, pc2):
    """ 
    Input:
      pc1, pc2: (Nx3) arrays where each row is a point in a point cloud
    Output:
      transform: (4,4) array representing the transformation from pc1 to pc2
    """
    H = np.zeros((3,3))

    # centroids - sum over the columns
    centroid1 = np.reshape(np.array(np.sum(pc1, axis=0))/ pc1.shape[0], (3,1))
    centroid2 = np.reshape(np.array(np.sum(pc2, axis=0))/ pc2.shape[0], (3,1))
        
    # H (3x3) 
    for i in range(pc1.shape[0]):
        pc1_point = np.reshape(pc1[i,:], (3,1))
        pc2_point = np.reshape(pc2[i,:], (3,1))
        H = H + np.dot( (pc1_point - centroid1) , np.transpose(pc2_point - centroid2)) 
    
    # Rotation, R (3x3)
    u, s, vh = np.linalg.svd(H) # u(9x9), s(6x1), vh(6x6)  
    v = vh.transpose()
    R = np.dot(v, np.transpose(u)) 
    # svd fix
    if np.linalg.det(R)<0:
        v[:,2] = -v[:,2]
        R = np.dot(v, np.transpose(u))
    
    # Translation, t (3x1)
    t = np.reshape(np.dot(-R, centroid1) + centroid2, (3,1))
    
    # Transformation, T
    transform = np.block([[R,t],
                         [0,0,0,1]])
    
    return transform


def Q1_I():
    """ Code for question 1i.
    Output:
      best_model_transform: (4,4) array representing the transformation of the object.
    """
    # note: 
        # - After iteration #1 of RANSAC the transformation will be identity 
        # - The goal is to find a model describing inliers from the given data set.
        
    # initialize constants
    prev_inliers_count = 0
    inliers_count = 3
    epsilon = 0.001
    
    # load point clouds and pixel ids
    pc1 = np.load('pointcloud1.npy') #Nx3
    pc2 = np.load('pointcloud2.npy') #Nx3
    id1 = np.load('pointcloud1_ids.npy')
    id2 = np.load('pointcloud2_ids.npy')
    
    # reorder point cloud pixels so that rows in pc1 and pc2 correspond to each other
    reordered_pc1 = np.zeros((pc1.shape[0], 3)) #Nx3
    for id,pt in zip(id1,pc1):
        reordered_pc1[id]=pt[:3]      
    reordered_pc2 = np.zeros((pc2.shape[0], 3)) #Nx3
    for id,pt in zip(id2,pc2):
        reordered_pc2[id]=pt[:3]
    
    # homogeneous coordinates and transpose
    pc1_ordered_h = np.block([[np.transpose(reordered_pc1)],
                              [np.ones((1,reordered_pc1.shape[0]))]]) #4xN 
    pc2_ordered_h = np.block([[np.transpose(reordered_pc2)],
                               [np.ones((1,reordered_pc2.shape[0]))]]) #4xN 
    
    # initialize group of 3 points we want to track
    pc1_newGroup = np.array([reordered_pc1[4,:], 
                             reordered_pc1[9,:], 
                             reordered_pc1[20,:]])
    pc2_newGroup = np.array([reordered_pc2[4,:], 
                             reordered_pc2[9,:], 
                             reordered_pc2[20,:]])
    
    ''' RANSAC loop:
        1) Compute homography H and corresponding transformation Tbs of a 
           random seed group of size 3 (the minimal number of points required 
           to estimate the parameters of the model, in this case to calculate H) 
        2) Find inlires to this model and add them to the group: ||p_b, Tbs*p_s|| < ε
        3) Re-compute H and T for this group 
        4) Keep model with the largest number of inlires
    '''
    while (inliers_count > prev_inliers_count):
        pc1_group = pc1_newGroup
        pc1_newGroup = np.array([])
        pc2_group = pc2_newGroup
        pc2_newGroup = np.array([])
        prev_inliers_count = inliers_count
        inliers_count = 0
        # 1. Transform (4x4)
        best_model_transform = find_transformation_for_variable_size_pc(pc1_group, pc2_group)
        # 2. Create (Nx3) group of inlires
        for i in range(pc2.shape[0]):
            if (np.linalg.norm(pc2_ordered_h[:,i] - np.dot(best_model_transform,np.transpose(pc1_ordered_h[:,i]))) < epsilon):
                inliers_count = inliers_count + 1 
                pc1_row_to_add = np.reshape(reordered_pc1[i,:], (1,3))
                pc2_row_to_add = np.reshape(reordered_pc2[i,:], (1,3))
                if (i == 0): 
                    pc1_newGroup = pc1_row_to_add
                    pc2_newGroup = pc2_row_to_add
                else:
                    pc1_newGroup = np.append(pc1_newGroup, pc1_row_to_add, axis=0)
                    pc2_newGroup = np.append(pc2_newGroup, pc2_row_to_add, axis=0)  
                    
    return best_model_transform


def Q1_J(transform):
    """ Code for question 1j.
    Input:
      transform: (4,4) numpy array. The result from Q1_H.
    Output:
      twist: a tuple (vx,vy,vz,wx,wy,wz).
    """
    # Note: 
        # pc1 is the pose of the "object" without any motion
        # pc2 after some motion that is the effect of integrating the twist
    
    # 1) T = e^[V] -> [V] = Ln(T): 
    time = 2 
    twist_skew = scipy.linalg.logm(transform)/time
    
    # 2) From skew symm. to vector:
    twist = (twist_skew[0,3], twist_skew[1,3],twist_skew[2,3],
             twist_skew[2,1], twist_skew[0,2], twist_skew[1,0])
    
    return twist


'''----------------- TEST CODE ------------------------------'''
'''----------------------------------------------------------'''

if __name__ == "__main__":
    quat_RF_WF = np.array([0.14,0.28,0.56,0.76])
    quat_RF_WF /= np.linalg.norm(quat_RF_WF)
    quat_RF_WF = tuple(quat_RF_WF)
    t_RF_WF = np.array([0.5, -9.2, 2.4])

    u_CF_RF = (1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0)
    t_CF_RF = np.array([0.1, -0.1, 1.2])

    transform_RF_WF_t0, transform_CF_RF_t0, transform_CF_WF_t0 = Q1_A(t_RF_WF, quat_RF_WF, t_CF_RF, u_CF_RF, np.pi/2.0)
    print("Q1_A: ")
    print(transform_CF_WF_t0)

    twist = (0.1, -0.02, 0.0, 0.0, 0.09, 0.1)
    delta_transform_RF_WF = Q1_B(twist, 2.0)
    print("Q1_B: ")
    print(delta_transform_RF_WF)

    twist = (0.0, 0.0, 0.0, 0.0, 0.0, 0.2)
    delta_transform_CF_RF = Q1_C(twist, 2.0)
    print("Q1_C: ")
    print(delta_transform_CF_RF)

    transform_CF_WF_t2 = Q1_D(transform_RF_WF_t0, transform_CF_RF_t0, delta_transform_RF_WF, delta_transform_CF_RF)
    print("Q1_D: ")
    print(transform_CF_WF_t2)

    pc1 = np.load('pointcloud1.npy')
    intrinsic = np.array([[486, 0, 318.5],
                          [0, 491, 237],
                          [0, 0, 1]])
    pc_position, pc_color = Q1_E(transform_CF_WF_t0, pc1)

    pc_position, pc_color = Q1_F(pc_position, pc_color, intrinsic)

    Q1_G(transform_CF_WF_t0, transform_CF_WF_t2, intrinsic)

    transform = Q1_H(4,9,20)
    print("Q1_H: ")
    print(transform)

    transform = Q1_I()
    print("Q1_I: ")
    print(transform)

    twist = Q1_J(transform)
    print("Q1_J: ")
    print(twist)
