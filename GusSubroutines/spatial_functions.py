import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------- 1. PEAKS (Estático y Dinámico) ----------------

def peaks(x, y):
    """Versión estándar estática de MATLAB."""
    return 3*(1-x)**2 * np.exp(-(x**2) - (y+1)**2) \
           - 10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) \
           - 1/3 * np.exp(-(x+1)**2 - y**2)

def peaks_dynamic(x, y, t):
    """Versión dinámica: los picos oscilan y rotan levemente."""
    x_rot = x * np.cos(0.1*t) - y * np.sin(0.1*t)
    y_rot = x * np.sin(0.1*t) + y * np.cos(0.1*t)
    amp_mod = np.sin(0.5 * t)
    return peaks(x_rot, y_rot) * amp_mod

# ---------------- 2. FLUIDO (Estático y Dinámico) ----------------

def fluid_static(X, Y, num_drops=5):
    """Versión estática: Interferencia de gotas instantánea."""
    eta = np.zeros_like(X)
    # Genera gotas aleatorias fijas
    seeds = [(0.4, 0.4), (-0.3, 0.2), (0.1, -0.5), (-0.4, -0.4), (0, 0)]
    for i in range(min(num_drops, len(seeds))):
        cx, cy = seeds[i]
        r = np.sqrt((X-cx)**2 + (Y-cy)**2)
        # Función de onda amortiguada
        eta += np.exp(-2*r) * np.cos(15*r)
    return eta

class FluidSimulator:
    """Clase para manejar la versión dinámica (Ecuación de onda)."""
    def __init__(self, N=500, L=2.0):
        self.N, self.L = N, L
        self.dx = L / N
        self.x = np.linspace(-L/2, L/2, N)
        self.X, self.Y = np.meshgrid(self.x, self.x)
        self.g, self.H, self.gamma = 9.81, 1.0, 5.0
        self.dt = 0.01
        self.eta = np.zeros((N, N))
        self.u = np.zeros((N, N))
        self.v = np.zeros((N, N))

    def step(self, t):
        # Diferencias finitas simplificadas
        def deriv(F, axis): return np.gradient(F, self.dx, axis=axis)
        
        self.u += self.dt * (-self.g * deriv(self.eta, 1) - self.gamma * self.u)
        self.v += self.dt * (-self.g * deriv(self.eta, 0) - self.gamma * self.v)
        self.eta -= self.dt * self.H * (deriv(self.u, 1) + deriv(self.v, 0))
        
        # Inserción de gota periódica
        if t % 50 == 0:
            cx, cy = 0.4*np.cos(t/100), 0.4*np.sin(t/100)
            self.eta += 0.5 * np.exp(-((self.X-cx)**2 + (self.Y-cy)**2) / 0.05)
        return self.eta

# ---------------- 3. EJEMPLO DE USO ----------------

N = 400
x = np.linspace(-3, 3, N)
X, Y = np.meshgrid(x, x)

# --- Visualización Estática ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Test Peaks
z_peaks = peaks(X, Y)
axs[0].imshow(z_peaks, cmap='viridis')
axs[0].set_title("Peaks Estático")

# Test Fluid
z_fluid = fluid_static(X/3, Y/3) # Escalado para el dominio
axs[1].imshow(z_fluid, cmap='RdBu')
axs[1].set_title("Fluido Estático (Snapshot)")

plt.show()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- (Aquí irían tus funciones peaks, peaks_dynamic y la clase FluidSimulator) ---

# Configuración inicial
N = 256
L = 4.0
x = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, x)

# Instanciamos el simulador de fluido
fluid_sim = FluidSimulator(N=N, L=L)

# Configuración de la figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
im_peaks = ax1.imshow(np.zeros((N, N)), cmap='viridis', extent=[-L/2, L/2, -L/2, L/2])
ax1.set_title("Peaks Dinámico")
im_fluid = ax2.imshow(np.zeros((N, N)), cmap='RdBu_r', extent=[-L/2, L/2, -L/2, L/2])
ax2.set_title("Fluido Dinámico (Wave Eq)")

def update(frame):
    # 1. Actualizar Peaks (Fase matemática pura)
    z_p = peaks_dynamic(X, Y, frame * 0.1)
    im_peaks.set_data(z_p)
    im_peaks.set_clim(z_p.min(), z_p.max())
    
    # 2. Actualizar Fluido (Física de ondas)
    # Hacemos 5 pasos de simulación por cada frame visual para mayor estabilidad
    for _ in range(5):
        z_f = fluid_sim.step(frame)
    
    im_fluid.set_data(z_f)
    vlimit = np.max(np.abs(z_f)) if np.max(np.abs(z_f)) > 0 else 1
    im_fluid.set_clim(-vlimit, vlimit)
    
    return im_peaks, im_fluid

ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.tight_layout()
plt.show()