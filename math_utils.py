import numpy as np

def Seidel(x, y, a, b, c, d, e, f, g):
    """Seidel polynomial"""
    s = (a * ((x**2) + (y**2))**2 + 
         b * ((x**2) + (y**2)) * y + 
         c * ((x**2) + (3 * (y**2))) + 
         d * ((x**2) + (y**2)) + 
         e * y + 
         f * x + 
         g)
    return s

def csc(sigma):
    """Cosecant function"""
    return 1 / np.sin(sigma)

def sec(sigma):
    """Secant function"""
    return 1 / np.cos(sigma)

def cot(sigma):
    """Cotangent function"""
    return np.cos(sigma) / np.sin(sigma)

def c_matx(n, m, error, shape):
    """Matrix C with error"""
    c = np.zeros(shape) + 1/4 * np.sinc(1/2 * n) * np.sinc(1/2 * m)
    return np.random.normal(c, c * error, shape)

def c1_matx(n, m, error, shape):
    """Matrix C1 with error"""
    c = np.zeros(shape) + 1/4 * np.sinc(1/2 * (n+1)) * np.sinc(1/2 * m)
    return np.random.normal(c, c * error, shape)

def least_squares(In, Qn):
    """Least squares solution"""
    A = np.linalg.inv(Qn.T @ Qn)
    B = In.T @ Qn
    U = B @ A
    return U.T

def c(n, m):
    """C function"""
    return 0.5 * np.sinc(0.5 * n) * 0.5 * np.sinc(0.5 * m)

def gradtorad(x):
    """Degrees to radians"""
    return x * np.pi / 180

def radtograd(r):
    """Radians to degrees"""
    return r * 180 / np.pi

def prom_filter(M):
    """Simple 3x3 average filter"""
    x, y = M.shape
    M_filter = np.zeros((x-1, y-1))
    
    for i in range(x-2):
        for j in range(y-2):
            Sum = 0
            for k in range(3):
                for l in range(3):
                    Sum += M[i+k, j+l]
                    if k == 2 and l == 2:
                        M_filter[i+1, j+1] = Sum / 9
    return M_filter[1:x-1, 1:y-1]

def calculate_phase_rms_error(original_matrix, recovered_matrix):
    """RMS error for phase matrices (handles 2π ambiguity)"""
    if original_matrix.shape != recovered_matrix.shape:
        raise ValueError("Las dimensiones de las matrices deben ser idénticas.")
    
    difference = original_matrix - recovered_matrix
    wrapped_difference = np.angle(np.exp(1j * difference))
    mean_squared_error = np.mean(wrapped_difference**2)
    return np.sqrt(mean_squared_error)

def calculate_rms_error(original_matrix, recovered_matrix):
    """RMS error for amplitude matrices"""
    difference = original_matrix - recovered_matrix
    mean_squared_error = np.mean(difference**2)
    return np.sqrt(mean_squared_error)