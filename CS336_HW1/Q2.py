import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# print(np.version.version)

def Q2_A():
    """ Code for question 2a.
    Output:
      p0, p1, p2: (N,2) numpy arrays representing the pixel coordinates of the tracked features.
      Include the visualization and your answer to the questions in the separate PDF.
    """  
    # Parameters for ShiTomasi corner detection
    feature_params = dict( maxCorners = 200,
                           qualityLevel = 0.03, #0.03
                           minDistance = 7,
                           blockSize = 7 )
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (75,75),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
    
    # Create some random colors
    color = np.random.randint(0,255,(200,3))
    
    # Take first frame and find corners in it
    old_img = cv.imread("rgb1.png", cv.IMREAD_COLOR )  #Read the image file
    old_gray = cv.cvtColor(old_img, cv.COLOR_BGR2GRAY)
    p0_temp = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    
    p0 = p0_temp
    # reshape from (200, 1, 2) to (200,2) --> each row is a point, P'[x_c, y_c]
    p0 = np.reshape(p0,(p0.shape[0],p0.shape[2])) # (Nx2) where N = 200
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_img)
    for img_file_name in ("rgb2.png", "rgb3.png"):
        frame = cv.imread(img_file_name, cv.IMREAD_COLOR ); 
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculate optical flow
        p1_temp, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_temp, None, **lk_params)
        # Select good points
        good_new = p1_temp[st==1]
        good_old = p0_temp[st==1]
        # Draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv.add(frame,mask)
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0_temp = good_new.reshape(-1,1,2)
        if img_file_name == "rgb2.png":
            p1 = p1_temp
            p1 = np.reshape(p1,(p1.shape[0],p1.shape[2])) # reshape from (200, 1, 2) to (200,2) --> each row is a point, P'[x_c, y_c]
        elif img_file_name == "rgb3.png":
            p2 = p1_temp
            p2 = np.reshape(p2,(p2.shape[0],p2.shape[2])) # reshape from (200, 1, 2) to (200,2) --> each row is a point, P'[x_c, y_c]
            plt.imshow(img)
            plt.title("Optical Flow (window size 75x75)")
            plt.show()
            plt.imsave("opticalFlow_75x75.png",img)

    return p0, p1, p2


def Q2_B(p0, p1, p2, intrinsic):
    """ Code for question 2b.
    Note that depth maps contain NaN values.
    Features that have NaN depth value in any of the frames should be excluded in the result.
    Input:
      p0, p1, p2: (N,2) numpy arrays, the results from Q2_A. The 2D coordinates of the tracked points on the image.
      intrinsic: (3,3) numpy array representing the camera intrinsic.
    Output:
      p0, p1, p2: (N,3) numpy arrays, the 3D positions of the tracked features in each frame.
    """
    depth0 = np.loadtxt('depth1.txt') #(480x640) we have depth values for all the pixels, but we are only tracking N = 200 of them
    depth1 = np.loadtxt('depth2.txt')
    depth2 = np.loadtxt('depth3.txt')
    
    p0_3d = np.zeros((200,3))
    p1_3d = np.zeros((200,3))
    p2_3d = np.zeros((200,3))
    
    # add z values to each point: P' --> Ph'[x_c*z, y_c*z, z], (Nx3)
    for i in range(p0.shape[0]):
        d0 = depth0[int(p0[i,1]),int(p0[i,0])]
        d1 = depth1[int(p1[i,1]),int(p1[i,0])]
        d2 = depth2[int(p2[i,1]),int(p2[i,0])]
        p0_3d[i,:] = [ p0[i,0]*d0, p0[i,1]*d0, d0]
        p1_3d[i,:] = [ p1[i,0]*d1, p1[i,1]*d1, d1]
        p2_3d[i,:] = [ p2[i,0]*d2, p2[i,1]*d2, d2]
    
    # remove points that have a NaN depth
    nanArray_p0 = np.logical_not(np.isnan(p0_3d[:,2])) # column with False on the isNaN depth values(Nx1) for t=0
    nanArray_p1 = np.logical_not(np.isnan(p1_3d[:,2])) 
    nanArray_p2 = np.logical_not(np.isnan(p2_3d[:,2]))
    nanArray_p01 = np.logical_and(nanArray_p0,nanArray_p1)
    nanArray = np.logical_and(nanArray_p01, nanArray_p2)
    p0 = p0_3d[nanArray,:] # keep only the rows corresponding to Trues in the nanArray at all time stamps
    p1 = p1_3d[nanArray,:]
    p2 = p2_3d[nanArray,:]

    # real world point: Ph' --> P[x,y,z] = k_inv * Ph'
    k_inv = np.linalg.inv(intrinsic)
    p0 = np.transpose(np.dot(k_inv,np.transpose(p0))) # (Nx3)
    p1 = np.transpose(np.dot(k_inv,np.transpose(p1)))
    p2 = np.transpose(np.dot(k_inv,np.transpose(p2)))    

    return p0, p1, p2


'''----------------- TEST CODE ------------------------------'''
'''----------------------------------------------------------'''

if __name__ == "__main__":
    p0, p1, p2 = Q2_A()  
    intrinsic = np.array([[486, 0, 318.5],
                          [0, 491, 237],
                          [0, 0, 1]])
    p0, p1, p2 = Q2_B(p0, p1, p2, intrinsic)
