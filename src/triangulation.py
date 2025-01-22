import numpy as np
from scipy.optimize import least_squares
import logging

def linear_triangulation(K, C1, R1, C2, R2, pts1, pts2):
    # Ensure C1 and C2 are column vectors
    C1 = C1.reshape(3, 1)
    C2 = C2.reshape(3, 1)

    I = np.identity(3)

    assert C1.shape == (3, 1), f"C1 shape is {C1.shape}, expected (3, 1)"
    assert C2.shape == (3, 1), f"C2 shape is {C2.shape}, expected (3, 1)"

    P1 = K @ R1 @ np.hstack((I, -C1))
    P2 = K @ R2 @ np.hstack((I, -C2))

    X = []
    for i in range(len(pts1)):
        x1 = pts1[i, 0]
        y1 = pts1[i, 1]
        x2 = pts2[i, 0]
        y2 = pts2[i, 1]

        A = np.array([
            y1 * P1[2, :] - P1[1, :],
            P1[0, :] - x1 * P1[2, :],
            y2 * P2[2, :] - P2[1, :],
            P2[0, :] - x2 * P2[2, :]
        ])

        _, _, VT = np.linalg.svd(A)
        X_i = VT[-1]
        X.append(X_i / X_i[3])

    return np.array(X)

def non_linear_triangulation(K, pts1, pts2, X_initial, R1, C1, R2, C2):
    # Ensure C1 and C2 are column vectors
    C1 = C1.reshape(3, 1)
    C2 = C2.reshape(3, 1)

    I = np.identity(3)
    P1 = K @ R1 @ np.hstack((I, -C1))
    P2 = K @ R2 @ np.hstack((I, -C2))

    if pts1.shape[0] != pts2.shape[0] or pts1.shape[0] != X_initial.shape[0]:
        raise ValueError("Check point dimensions in non_linear_triangulation")

    x3D_optimized = []
    for i in range(len(X_initial)):
        X0 = X_initial[i][:3]  # Initial guess for optimization

        optimized_params = least_squares(
            fun=reprojection_loss,
            x0=X0,
            method="trf",
            args=[pts1[i], pts2[i], P1, P2],
        )
        X1 = optimized_params.x
        X1_homogeneous = np.append(X1, 1)
        x3D_optimized.append(X1_homogeneous)
    return np.array(x3D_optimized)

def reprojection_loss(X, pt1, pt2, P1, P2):
    X_h = np.append(X, 1)

    x_proj1 = P1 @ X_h
    x_proj1 /= x_proj1[2]
    u1, v1 = pt1
    E1 = np.square(u1 - x_proj1[0]) + np.square(v1 - x_proj1[1])

    x_proj2 = P2 @ X_h
    x_proj2 /= x_proj2[2]
    u2, v2 = pt2
    E2 = np.square(u2 - x_proj2[0]) + np.square(v2 - x_proj2[1])

    return E1 + E2