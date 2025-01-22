import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from src.build_visibility_matrix import get_camera_point_indices, get_2d_points
from utils.logging_utils import log_time
import logging
import cv2

@log_time
def bundle_adjustment(
    X_index,
    visibility_matrix,
    X_all,
    X_found,
    feature_x,
    feature_y,
    filtered_feature_flag,
    R_set,
    C_set,
    K,
    nCam,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger(__name__)

    number_of_cam = nCam + 1
    points_3d = X_all[X_index]
    points_2d = get_2d_points(X_index, visibility_matrix, feature_x, feature_y)

    camera_params = []
    for i in range(number_of_cam):
        R = R_set[i]
        C = C_set[i].flatten()
        rvec, _ = cv2.Rodrigues(R)
        params = np.hstack((rvec.flatten(), C))
        camera_params.append(params)
    camera_params = np.array(camera_params).reshape(-1)

    x0 = np.hstack((camera_params, points_3d.flatten()))
    n_cameras = number_of_cam
    n_points = points_3d.shape[0]

    camera_indices, point_indices = get_camera_point_indices(visibility_matrix)

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    res = least_squares(
        bundle_adjustment_func,
        x0,
        jac_sparsity=A,
        verbose=2,
        x_scale='jac',
        ftol=1e-4,
        method='trf',
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K)
    )

    x = res.x
    camera_params = x[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = x[n_cameras * 6:].reshape((n_points, 3))

    # Update camera poses
    R_set_new = []
    C_set_new = []
    for params in camera_params:
        rvec = params[:3]
        C = params[3:].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        R_set_new.append(R)
        C_set_new.append(C)

    # Update 3D points
    X_all_new = X_all.copy()
    X_all_new[X_index] = points_3d

    return R_set_new, C_set_new, X_all_new

def bundle_adjustment_func(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    return (points_proj - points_2d).ravel()

def project(points_3d, camera_params, K):
    points_proj = []
    for point, params in zip(points_3d, camera_params):
        rvec = params[:3]
        tvec = params[3:]
        R, _ = cv2.Rodrigues(rvec)
        point_proj = (K @ (R @ point.reshape(3, 1) + tvec.reshape(3, 1))).flatten()
        point_proj /= point_proj[2]
        points_proj.append(point_proj[:2])
    return np.array(points_proj)

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A