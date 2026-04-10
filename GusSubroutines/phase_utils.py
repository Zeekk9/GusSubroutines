import numpy as np
from scipy.fftpack import dct, idct

def wrap_to_pi(x):
    """Wrap phase to [-π, π]"""
    return (x + np.pi) % (2 * np.pi) - np.pi

def solve_poisson(rho):
    """Solve Poisson equation using DCT"""
    N, M = rho.shape
    dct_rho = dct(dct(rho.T, norm='ortho').T, norm='ortho')
    I, J = np.meshgrid(np.arange(M), np.arange(N))
    denom = 2 * (np.cos(np.pi * I / M) + np.cos(np.pi * J / N) - 2)
    denom[0, 0] = 1  # Avoid division by zero
    dct_phi = dct_rho / denom
    dct_phi[0, 0] = 0  # Set the mean to zero
    phi = idct(idct(dct_phi.T, norm='ortho').T, norm='ortho')
    return phi

def apply_q(p, ww):
    """Apply Q operator"""
    dx = np.concatenate([np.diff(p, axis=1), np.zeros((p.shape[0], 1))], axis=1)
    dy = np.concatenate([np.diff(p, axis=0), np.zeros((1, p.shape[1]))], axis=0)

    ww_dx = ww * dx
    ww_dy = ww * dy

    ww_dx2 = np.concatenate([np.zeros((p.shape[0], 1)), ww_dx], axis=1)
    ww_dy2 = np.concatenate([np.zeros((1, p.shape[1])), ww_dy], axis=0)

    q_p = np.diff(ww_dx2, axis=1) + np.diff(ww_dy2, axis=0)
    return q_p

def phase_unwrap(psi, weight=None):
    """2D phase unwrapping"""
    if weight is None:
        dx = np.concatenate([np.zeros((psi.shape[0], 1)), wrap_to_pi(np.diff(psi, axis=1)), np.zeros((psi.shape[0], 1))], axis=1)
        dy = np.concatenate([np.zeros((1, psi.shape[1])), wrap_to_pi(np.diff(psi, axis=0)), np.zeros((1, psi.shape[1]))], axis=0)
        rho = np.diff(dx, axis=1) + np.diff(dy, axis=0)
        phi = solve_poisson(rho)
    else:
        if psi.shape != weight.shape:
            raise ValueError("Weight must be the same shape as the input phase")
        
        dx = np.concatenate([wrap_to_pi(np.diff(psi, axis=1)), np.zeros((psi.shape[0], 1))], axis=1)
        dy = np.concatenate([wrap_to_pi(np.diff(psi, axis=0)), np.zeros((1, psi.shape[1]))], axis=0)

        ww = weight ** 2
        ww_dx = ww * dx
        ww_dy = ww * dy

        ww_dx2 = np.concatenate([np.zeros((psi.shape[0], 1)), ww_dx], axis=1)
        ww_dy2 = np.concatenate([np.zeros((1, psi.shape[1])), ww_dy], axis=0)

        rk = np.diff(ww_dx2, axis=1) + np.diff(ww_dy2, axis=0)
        norm_r0 = np.linalg.norm(rk)
        phi = np.zeros_like(psi)
        eps = 1e-8
        k = 0

        while np.any(rk != 0):
            zk = solve_poisson(rk)
            if k == 0:
                pk = zk
            else:
                beta_k = np.sum(rk * zk) / np.sum(rk_prev * zk_prev)
                pk = zk + beta_k * pk

            rk_prev = rk
            zk_prev = zk

            q_pk = apply_q(pk, ww)
            alpha_k = np.sum(rk * zk) / np.sum(pk * q_pk)

            phi += alpha_k * pk
            rk -= alpha_k * q_pk

            if k >= psi.size or np.linalg.norm(rk) < eps * norm_r0:
                break
            k += 1
    return phi

def itoh_2D(W):
    """Itoh 2D phase unwrapping"""
    renglon, columna = W.shape
    phi = np.zeros(W.shape)
    psi = np.zeros(W.shape)
    phi[0, 0] = W[0, 0]
    
    # Desenvolver primera columna
    for m in range(1, columna):
        Delta = W[0, m] - W[0, m - 1]
        WDelta = np.arctan2(np.sin(Delta), np.cos(Delta))
        phi[0, m] = phi[0, m - 1] + WDelta
    psi[0, :] = phi[0, :]

    for k in range(columna):
        psi[0, k] = W[0, k]
        for p in range(1, renglon):
            Delta = W[p, k] - W[p - 1, k]
            WDelta = np.arctan2(np.sin(Delta), np.cos(Delta))
            phi[p, k] = phi[p - 1, k] + WDelta
    return phi

def wrap(W):
    """Wrap phase to [-π, π]"""
    return np.arctan2(np.sin(W), np.cos(W))

def simple_align(phi_rec, phi_ref):
    """
    Encuentra automáticamente si la fase está invertida y el offset constante.
    """
    errors = []
    for sign in [1, -1]:
        # Calculamos la diferencia compleja para obtener el offset promedio
        # diff_exp = exp(i * ref) / exp(i * sign * rec)
        diff_complex = np.exp(1j * phi_ref) / np.exp(1j * (sign * phi_rec))
        offset = np.angle(np.mean(diff_complex))

        # Aplicamos el candidato
        phi_cand = np.arctan2(np.sin(sign * phi_rec + offset),
                              np.cos(sign * phi_rec + offset))

        # Error cuadrático medio circular
        error = np.mean(1 - np.cos(phi_ref - phi_cand))
        errors.append((error, phi_cand, offset, sign))

    # Seleccionamos la combinación con menor error
    best = min(errors, key=lambda x: x[0])
    return best[1]

def align_offset(phi_recovered, phi_theoretical):
    """
    Alinea la fase recuperada con la fase teórica removiendo el offset constante.
    Asume que el desplazamiento es constante en toda la imagen.
    """
    # Calcula el offset promedio entre las dos fases
    offset = np.mean(phi_recovered - phi_theoretical)

    # Remueve el offset
    phi_aligned = phi_recovered - offset

    return phi_aligned


def align_phase_fast(phi_rec, phi_ref):
    """
    Alineación ultra-rápida basada en estadística circular.
    Calcula el offset óptimo analíticamente sin iteraciones.
    """
    # 1. Representación compleja de la diferencia: exp(i * (ref - rec))
    # Esto extrae el desfase relativo pixel a pixel
    delta_complex = np.exp(1j * phi_ref) * np.conj(np.exp(1j * phi_rec))
    
    # 2. El ángulo del promedio de todos los fasores es el offset global
    offset = np.angle(np.mean(delta_complex))
    
    # 3. Aplicar el offset y re-envolver
    return np.arctan2(np.sin(phi_rec + offset), np.cos(phi_rec + offset))


def align_phase_simple(phi_rec, phi_ref):
    """
    Alineación analítica usando el promedio de la diferencia compleja.
    Calcula el offset exacto y detecta el signo automáticamente.
    """
    def get_best_offset(p_rec, p_ref):
        # Representación compleja de la diferencia: exp(i * (ref - rec))
        diff_complex = np.exp(1j * p_ref) / np.exp(1j * p_rec)
        # El ángulo del promedio es el offset óptimo (piston)
        offset = np.angle(np.mean(diff_complex))
        return offset

    # Probar signo positivo
    off_pos = get_best_offset(phi_rec, phi_ref)
    phi_pos = np.arctan2(np.sin(phi_rec + off_pos), np.cos(phi_rec + off_pos))
    err_pos = np.mean(1 - np.cos(phi_ref - phi_pos))

    # Probar signo negativo
    off_neg = get_best_offset(-phi_rec, phi_ref)
    phi_neg = np.arctan2(np.sin(-phi_rec + off_neg), np.cos(-phi_rec + off_neg))
    err_neg = np.mean(1 - np.cos(phi_ref - phi_neg))

    # Retornar la que minimice el error circular
    return phi_pos if err_pos < err_neg else phi_neg


def align_phase_robust(phi_rec, phi_ref, steps=360):
    """
    Alineación por fuerza bruta. Útil para validar resultados 
    cuando el signo o el offset son muy erráticos.
    """
    best_phi = np.copy(phi_rec)
    min_error = np.inf
    
    # Probamos ambos signos (fase directa e invertida)
    for sign in [1, -1]:
        offsets = np.linspace(-np.pi, np.pi, steps)
        for off in offsets:
            # Aplicamos transformación circular
            phi_cand = np.arctan2(np.sin(sign * phi_rec + off), 
                                  np.cos(sign * phi_rec + off))
            
            # Error basado en la distancia cordal (robusto a saltos de 2pi)
            error = np.mean(1 - np.cos(phi_ref - phi_cand))
            
            if error < min_error:
                min_error = error
                best_phi = phi_cand
                
    return best_phi



def apply_phase_noise(data, t=0, noise_lvl=0.0, drift_lvl=0.0, seed=None):
    N = data.shape[0]
    if seed is not None:
        np.random.seed(seed)

    noise_lvl_percent = noise_lvl/100
    drift_lvl_percent = drift_lvl/100

    # Si la data es 1D (un vector de píxeles)
    if data.ndim == 1:
        white_noise = np.random.normal(0, noise_lvl_percent, N)
        x = np.linspace(-1, 1, N)
        drift_lvl_percent = np.random.normal(0, noise_lvl_percent)
        drift = drift_lvl_percent * np.sin(x + t*0.01)
    else:
        # Si es 2D (la imagen completa)
        white_noise = np.random.normal(0, noise_lvl_percent, (N, N))
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, x)
        drift = drift_lvl_percent * np.sin(X + t*0.1) * np.cos(Y - t*0.05)

    return data + white_noise + drift