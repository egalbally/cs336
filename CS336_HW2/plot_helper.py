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
    raise NotImplementedError()
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

    raise NotImplementedError()
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
