import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Ellipse

def draw_2d(ax, covariance, mean):
    """ Implement function to plot one ellipse on ax.
    Input:
      ax: the plt axis object.
      covariance: (2,2) numpy array, The predicted observation covariance you obtain from EKF.
      mean: (2,) numpy array. The mean predicted observation you obtain from EKF.
    Note:
      calculate H, W, A, three floating numbers that will be used in the plt Ellipse.
    """
    # find eigenvalues and order from big to small
    eigenValues, eigenVectors = np.linalg.eig(covariance)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    # plot vars
    confidence_95 = 2*np.sqrt(5.991)
    H = confidence_95*np.sqrt(eigenValues[0])
    W = confidence_95*np.sqrt(eigenValues[1])
    A = np.arctan(eigenVectors[1,0]/eigenVectors[0,0])
    el = Ellipse(xy=mean, width=W, height=H, angle=A,
                 color='C1', alpha=0.5)
    ax.add_artist(el)


def draw_3d(ax, covariance, mean):
    """ Implement function to plot one ellipsoid on 3d ax.
    Input:
      ax: the plt axis object.
      covariance: (3,3) numpy array. The filtered covariance of positions.
      mean: (3,) numpy array, The filtered mean of positions.
    Note:
      calculate pts, a (U, V, 3) numpy array of float, representing the x,y,z of
                     points on the ellipsoid. U and V represent the resolution in longitude and latitude.
                     refer to https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#surface-plots.
    """   
    eig_value, eig_vector = np.linalg.eig(covariance)
    idx = eig_value.argsort()[::-1]   
    eig_value = eig_value[idx]
    eig_vector = eig_vector[:,idx]
    u = np.linspace(0.0, 2.0 * np.pi, 10)
    v = np.linspace(0.0, np.pi, 10)
    pts = np.zeros((len(u), len(v),3))
    pts[:,:,0] = np.outer(np.cos(u), np.sin(v))
    pts[:,:,1] = np.outer(np.sin(u), np.sin(v))
    pts[:,:,2] = np.outer(np.ones_like(u), np.cos(v))
    pts = pts.dot(np.diag(2*np.sqrt(5.991*eig_value))).dot(eig_vector.transpose()) + np.tile(mean,((len(u), len(v), 1)))
    ax.plot_surface(pts[:,:,0], pts[:,:,1], pts[:,:,2], rstride=1, cstride=1,
                        color='C1', linewidth=0.1, alpha=0.5, shade=True)

if __name__ == "__main__":
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    covariance = np.array([[9.0,-4.0,2.0],
                           [-4.0,4.0,0],
                           [2.0,0,1.0]])
    center = np.zeros(3)
    draw_3d(ax, covariance, center)
    plt.show()
