import numpy as np
import logging

def extract_camera_poses(E):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R_set = []
    C_set = []

    R_set.append(U @ W @ Vt)
    R_set.append(U @ W @ Vt)
    R_set.append(U @ W.T @ Vt)
    R_set.append(U @ W.T @ Vt)
    C_set.append(U[:, 2])
    C_set.append(-U[:, 2])
    C_set.append(U[:, 2])
    C_set.append(-U[:, 2])

    for i in range(4):
        if np.linalg.det(R_set[i]) < 0:
            R_set[i] = -R_set[i]
            C_set[i] = -C_set[i]

    return R_set, C_set

def disambiguate_camera_pose(R_set, C_set, X_sets):
    best_i = -1
    max_positive_depths = 0

    for i in range(len(R_set)):
        R = R_set[i]
        C = C_set[i].reshape(3, 1)
        X = X_sets[i]
        X = X / X[:, 3].reshape(-1, 1)
        X = X[:, :3]

        r3 = R[2, :].reshape(1, -1)
        n_positive_depths = 0
        for X_i in X:
            X_i = X_i.reshape(3, 1)
            if (r3 @ (X_i - C)) > 0 and X_i[2] > 0:
                n_positive_depths += 1

        if n_positive_depths > max_positive_depths:
            best_i = i
            max_positive_depths = n_positive_depths

    R_best = R_set[best_i]
    C_best = C_set[best_i].reshape(3, 1)
    X_best = X_sets[best_i]
    return R_best, C_best, X_best