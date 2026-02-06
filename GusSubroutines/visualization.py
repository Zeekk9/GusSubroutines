import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.ticker as ticker

def progress_bar(current, total, prefix=''):
    """Imprime una actualización en la misma línea."""
    msg = f"\r{prefix} {current}/{total} frames..."
    sys.stdout.write(msg)
    sys.stdout.flush()

def matshow(position, mat, title):
    """Display matrix with colorbar"""
    plt.subplot(position)
    plt.imshow(mat, cmap='gist_heat')
    plt.title(title, fontsize=30)
    plt.axis('off')
    plt.colorbar()
    
def Plotting(fig, position, matrix, title, colormap, cbartitle, 
             show_xticks=True, show_yticks=True, phiwrap=False, 
             customlim=None, titlesize=35): # Usamos None por defecto
    
    # 1. Manejo de la posición (Soporta 341 o (3, 4, 10))
    if isinstance(position, tuple):
        ax = fig.add_subplot(*position)
    else:
        ax = fig.add_subplot(position)
    
    # 2. Lógica de límites (Custom vs Automático)
    if customlim is not None:
        # customlim debe ser una lista o tupla: [vmin, vmax]
        vmin, vmax = customlim
        im = ax.imshow(matrix, cmap=colormap, vmin=vmin, vmax=vmax)
    else:
        # Si es False o None, escala automáticamente a los datos de la matriz
        im = ax.imshow(matrix, cmap=colormap)
        
    ax.set_title(title, fontsize=titlesize)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.tick_params(axis='both', labelsize=15, length=0)

    if not show_xticks:
        ax.set_xticks([])
    if not show_yticks:
        ax.set_yticks([])
        

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbartitle, rotation=270, labelpad=20, size=20)
    # Solo cambiamos los ticks si phiwrap es True
    if phiwrap:
        cbar.ax.tick_params(labelsize=25, length=0)
        cbar.set_ticks([-np.pi, 0, np.pi])
        cbar.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    else:
        # Configuración estándar para los demás
        cbar.ax.tick_params(labelsize=15, length=0)
        cbar.locator = ticker.MaxNLocator(nbins=4)
        cbar.update_ticks()

    ax.grid(False)
    return ax


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