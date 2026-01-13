import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def matshow(position, mat, title):
    """Display matrix with colorbar"""
    plt.subplot(position)
    plt.imshow(mat, cmap='gist_heat')
    plt.title(title, fontsize=30)
    plt.axis('off')
    plt.colorbar()

def surf(p1, p2, p3, W, title):
    """3D surface plot"""
    fig = plt.figure(1)
    xlim = W[0, :].size
    ylim = W[:, 0].size
    x = np.linspace(0, xlim, xlim)
    y = np.linspace(0, ylim, ylim)
    X, Y = np.meshgrid(x, y)
    
    ax = fig.add_subplot(p1, p2, p3, projection='3d')
    plt.title(title, fontsize=30)
    ax.plot_surface(X, Y, W, cmap='gist_heat')
    plt.axis('on')

def show():
    """Maximize plot window based on OS"""
    if sys.platform.startswith('win'):
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
    elif sys.platform.startswith('linux'):
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
    elif sys.platform.startswith('darwin'):  # macOS
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
    else:
        print("Sistema operativo no soportado para maximización.")
    
    plt.show()