import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import dctn, idctn

def phase_unwrap(psi, weight=None):

    def wrapToPi(angle):
        return np.mod(angle + np.pi, 2 * np.pi) - np.pi

    def solve_poisson(rho):
        dct_rho = dctn(rho)
        N, M = rho.shape
        I, J = np.meshgrid(np.arange(M), np.arange(N))
        val_aux = np.nan_to_num(
            (2 * (np.cos(np.pi * I / M) + np.cos(np.pi * J / N) - 2)))
        dct_phi = np.nan_to_num(dct_rho / val_aux)
        dct_phi[0, 0] = 0  # Handling the inf/nan value
        phi = idctn(dct_phi)
        return phi

    def apply_Q(p, WW):
        dx = np.diff(p, axis=1)
        dx = np.concatenate((dx, np.zeros((p.shape[0], 1))), axis=1)

        dy = np.diff(p, axis=0)
        dy = np.concatenate((dy, np.zeros((1, p.shape[1]))), axis=0)

        WWdx = WW * dx
        WWdy = WW * dy

        WWdx2 = np.concatenate((np.zeros((p.shape[0], 1)), WWdx), axis=1)
        WWdy2 = np.concatenate((np.zeros((1, p.shape[1])), WWdy), axis=0)

        Qp = np.diff(WWdx2, axis=1) + np.diff(WWdy2, axis=0)
        return Qp

    if weight is None:  # Unweighted phase unwrap

        dx = np.concatenate((np.zeros((psi.shape[0], 1)), wrapToPi(
            np.diff(psi, axis=1)), np.zeros((psi.shape[0], 1))), axis=1)
        dy = np.concatenate((np.zeros((1, psi.shape[1])), wrapToPi(
            np.diff(psi, axis=0)), np.zeros((1, psi.shape[1]))), axis=0)

        rho = np.diff(dx, axis=1) + np.diff(dy, axis=0)
        phi = solve_poisson(rho)

    else:  # Weighted phase unwrap
        if weight.shape != psi.shape:
            raise ValueError(
                "Size of the weight must be the same as size of the wrapped phase")

        dx = np.concatenate((wrapToPi(np.diff(psi, axis=1)),
                            np.zeros((psi.shape[0], 1))), axis=1)
        dy = np.concatenate((wrapToPi(np.diff(psi, axis=0)),
                            np.zeros((1, psi.shape[1]))), axis=0)

        WW = weight * weight
        WWdx = WW * dx
        WWdy = WW * dy

        WWdx2 = np.concatenate((np.zeros((psi.shape[0], 1)), WWdx), axis=1)
        WWdy2 = np.concatenate((np.zeros((1, psi.shape[1])), WWdy), axis=0)

        rk = np.diff(WWdx2, axis=1) + np.diff(WWdy2, axis=0)
        norm_R0 = np.linalg.norm(rk)

        eps = 1e-8
        k = 0
        phi = np.zeros_like(psi)
        while not np.all(rk == 0):
            zk = solve_poisson(rk)
            k += 1

            if k == 1:
                pk = zk
            else:
                betak = np.sum(np.sum(rk * zk)) / \
                    np.sum(np.sum(rkprev * zkprev))
                pk = zk + betak * pk

            rkprev = rk
            zkprev = zk

            Qpk = apply_Q(pk, WW)
            alphak = np.sum(np.sum(rk * zk)) / np.sum(np.sum(pk * Qpk))
            phi += alphak * pk
            rk -= alphak * Qpk

            if k >= np.prod(psi.shape) or np.linalg.norm(rk) < eps * norm_R0:
                break

    return phi
