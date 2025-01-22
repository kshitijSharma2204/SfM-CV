import numpy as np
from scipy.optimize import least_squares
import cv2
import logging

def pnp_ransac(K, pts_2d, pts_3d, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    # Convert to proper format for cv2.solvePnPRansac
    pts_3d = pts_3d.astype(np.float64)
    pts_2d = pts_2d.astype(np.float64)

    dist_coeffs = np.zeros((4, 1))
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d,
        pts_2d,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        R, _ = cv2.Rodrigues(rvec)
        C = -R.T @ tvec
        logger.info(f"PnP RANSAC succeeded with {len(inliers)} inliers")
        return R, C.flatten()
    else:
        logger.error("PnP RANSAC failed")
        return None, None

def non_linear_pnp(K, pts_2d, pts_3d, R_init, C_init):
    X0 = np.hstack((cv2.Rodrigues(R_init)[0].flatten(), C_init))

    res = least_squares(
        pnp_reprojection_error,
        X0,
        args=(K, pts_2d, pts_3d),
        method='lm'
    )

    rvec = res.x[:3]
    C_opt = res.x[3:]
    R_opt, _ = cv2.Rodrigues(rvec)
    return R_opt, C_opt

def pnp_reprojection_error(X, K, pts_2d, pts_3d):
    rvec = X[:3]
    tvec = X[3:]
    R, _ = cv2.Rodrigues(rvec)
    pts_proj = (K @ (R @ pts_3d.T + tvec.reshape(3, 1))).T
    pts_proj = pts_proj[:, :2] / pts_proj[:, 2].reshape(-1, 1)
    error = (pts_2d - pts_proj).flatten()
    return error