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
    
def Plotting(fig, position, matrix, title, colormap, cbartitle=None, 
             show_xticks=True, show_yticks=True, phiwrap=False, 
             customlim=None, titlesize=35, 
             xlabel=None, ylabel=None, labelsize=20, tickssize=18,
             show_cbar=True, cbarlabelsize=15,cbartitlesize=20): # Agregamos el parámetro por default True
    
    if isinstance(position, tuple):
        ax = fig.add_subplot(*position)
    else:
        ax = fig.add_subplot(position)
    
    if customlim is not None:
        vmin, vmax = customlim
        im = ax.imshow(matrix, cmap=colormap, vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(matrix, cmap=colormap)
        
    ax.set_title(title, fontsize=titlesize)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=labelsize)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.tick_params(axis='both', labelsize=tickssize, length=0)

    if not show_xticks:
        ax.set_xticks([])
    if not show_yticks:
        ax.set_yticks([])
        

    if show_cbar:
        cbar = fig.colorbar(im, ax=ax)
        
        if phiwrap:
            cbar.set_ticks([-np.pi, 0, np.pi])
            cbar.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
            cbar.ax.tick_params(labelsize=cbarlabelsize, length=0, pad=3) 
            cbar.set_label(cbartitle, rotation=270, labelpad=12, size=cbartitlesize)
        else:
            cbar.ax.tick_params(labelsize=cbarlabelsize, length=0)
            cbar.locator = ticker.MaxNLocator(nbins=4)
            cbar.update_ticks()
            cbar.set_label(cbartitle, rotation=270, labelpad=20, size=cbartitlesize)
    # -------------------------------------

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
    
def save_to_vtk(matrix, filename, data_name="ScalarField", dx=1.0, dy=1.0):
    """
    Exporta una matriz de datos 2D (imágenes, fases, interferogramas) a formato VTK
    para ser visualizada y procesada en 3D dentro de ParaView.
    """
    if not filename.lower().endswith(('.vtk', '.vts')):
        filename_base = filename
    else:
        filename_base = filename.rsplit('.', 1)[0]

    # Intentar usar el método binario rápido (pyevtk)
    try:
        from pyevtk.hl import gridToVTK
        
        # Orientación correcta para ParaView
        matrix_ready = np.flipud(matrix)
        
        ny, nx = matrix_ready.shape
        nz = 1 
        
        # Crear mallas espaciales
        x = np.linspace(0, (nx - 1) * dx, nx)
        y = np.linspace(0, (ny - 1) * dy, ny)
        z = np.zeros(nz)
        
        X = np.zeros((nx, ny, nz))
        Y = np.zeros((nx, ny, nz))
        Z = np.zeros((nx, ny, nz))
        
        for i in range(nx):
            for j in range(ny):
                X[i, j, 0] = x[i]
                Y[i, j, 0] = y[j]
                Z[i, j, 0] = z[0]
                
        # Adaptar dimensiones al estándar VTK (Eje Z dummy)
        data_3d = matrix_ready.T[:, :, np.newaxis]
        
        # SOLUCIÓN ESTABLE: Forzar paso contiguo por memoria RAM a todas las matrices
        X = np.ascontiguousarray(X)
        Y = np.ascontiguousarray(Y)
        Z = np.ascontiguousarray(Z)
        data_3d = np.ascontiguousarray(data_3d)
        
        # Guardar archivo estructurado XML binario (.vts)
        gridToVTK(filename_base, X, Y, Z, pointData={data_name: data_3d})
        print(f"✅ Archivo estructurado binario guardado como: {filename_base}.vts")

    # Método alternativo nativo (ASCII .vtk) si pyevtk no está mapeado
    except ImportError:
        matrix_ready = np.flipud(matrix)
        ny, nx = matrix_ready.shape
        
        if not filename.endswith('.vtk'):
            filename = filename_base + '.vtk'
            
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"Exported from GusSubroutines - {data_name}\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {nx} {ny} 1\n")
            f.write(f"ORIGIN 0 0 0\n")
            f.write(f"SPACING {dx} {dy} 1.0\n")
            f.write(f"POINT_DATA {nx * ny}\n")
            f.write(f"SCALARS {data_name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            np.savetxt(f, matrix_ready.flatten(), fmt='%f')
            
        print(f"⚠️ Guardado en formato ASCII heredado como: {filename}")