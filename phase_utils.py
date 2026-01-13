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