import numpy as np

def get_observations_index_and_viz_mat(X_found, filtered_feature_flag, nCam):
    bin_temp = np.zeros(filtered_feature_flag.shape[0], dtype=int)
    for n in range(nCam + 1):
        bin_temp |= filtered_feature_flag[:, n]

    X_index = np.where(X_found.reshape(-1) & bin_temp)
    visibility_matrix = X_found[X_index].reshape(-1, 1)
    for n in range(nCam + 1):
        visibility_matrix = np.hstack(
            (visibility_matrix, filtered_feature_flag[X_index, n].reshape(-1, 1))
        )

    return X_index[0], visibility_matrix[:, 1:]

def get_2d_points(X_index, visibility_matrix, feature_x, feature_y):
    pts2D = []
    visible_feature_x = feature_x[X_index]
    visible_feature_y = feature_y[X_index]
    h, w = visibility_matrix.shape
    for i in range(h):
        for j in range(w):
            if visibility_matrix[i, j] == 1:
                pt = [visible_feature_x[i, j], visible_feature_y[i, j]]
                pts2D.append(pt)
    return np.array(pts2D).reshape(-1, 2)

def get_camera_point_indices(visibility_matrix):
    camera_indices = []
    point_indices = []
    h, w = visibility_matrix.shape
    for i in range(h):
        for j in range(w):
            if visibility_matrix[i, j] == 1:
                camera_indices.append(j)
                point_indices.append(i)
    return np.array(camera_indices), np.array(point_indices)