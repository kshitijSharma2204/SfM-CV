import numpy as np

def compute_essential_matrix(K, F):
    E = K.T @ F @ K
    U, S, Vt = np.linalg.svd(E)
    S = [1, 1, 0]
    E = U @ np.diag(S) @ Vt
    return E