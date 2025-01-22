import numpy as np
import logging

def normalize_points(uv):
    uv_mean = np.mean(uv, axis=0)
    u_mean, v_mean = uv_mean[0], uv_mean[1]
    u_cap, v_cap = uv[:, 0] - u_mean, uv[:, 1] - v_mean

    s = np.sqrt(2 / np.mean(u_cap**2 + v_cap**2))
    T_scale = np.diag([s, s, 1])
    T_trans = np.array([[1, 0, -u_mean], [0, 1, -v_mean], [0, 0, 1]])
    T = T_scale @ T_trans

    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T @ x_.T).T
    return x_norm, T

def estimate_fundamental_matrix(pts1, pts2):
    x1, x2 = pts1, pts2

    if x1.shape[0] > 7:
        x1_norm, T1 = normalize_points(x1)
        x2_norm, T2 = normalize_points(x2)

        A = np.zeros((len(x1_norm), 9))
        for i in range(len(x1_norm)):
            x1_i, y1_i = x1_norm[i][0], x1_norm[i][1]
            x2_i, y2_i = x2_norm[i][0], x2_norm[i][1]
            A[i] = [
                x1_i * x2_i,
                x2_i * y1_i,
                x2_i,
                y2_i * x1_i,
                y2_i * y1_i,
                y2_i,
                x1_i,
                y1_i,
                1,
            ]

        _, _, VT = np.linalg.svd(A)
        F = VT[-1].reshape(3, 3)

        U, S, Vt = np.linalg.svd(F)
        S = np.diag(S)
        S[2, 2] = 0
        F = U @ S @ Vt

        F = T2.T @ F @ T1
        F = F / F[2, 2]
        return F
    else:
        return None

def error_fundamental(pts1, pts2, F):
    x1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    x2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    errors = np.abs(np.sum(x2 * (F @ x1.T).T, axis=1))
    return errors

def get_inliers_ransac(pts1, pts2, idx, threshold=0.002, iterations=2000, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    max_inliers = []
    best_F = None

    for _ in range(iterations):
        sample_indices = np.random.choice(len(pts1), 8, replace=False)
        pts1_sample = pts1[sample_indices]
        pts2_sample = pts2[sample_indices]

        F = estimate_fundamental_matrix(pts1_sample, pts2_sample)
        if F is not None:
            errors = error_fundamental(pts1, pts2, F)
            inliers = np.where(errors < threshold)[0]
            if len(inliers) > len(max_inliers):
                max_inliers = inliers
                best_F = F

    if best_F is not None:
        return best_F, max_inliers
    else:
        logger.warning("Could not find a valid fundamental matrix")
        return None, []