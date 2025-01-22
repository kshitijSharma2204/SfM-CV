import os
import argparse
import numpy as np
import cv2

from src.feature_extraction import extract_features
from src.fundamental_matrix_estimation import estimate_fundamental_matrices
from src.essential_matrix_estimation import compute_essential_matrix
from src.camera_pose_estimation import (
    extract_camera_poses,
    disambiguate_camera_pose,
)
from src.triangulation import (
    linear_triangulation,
    non_linear_triangulation,
)
from src.pnp_estimation import (
    pnp_ransac,
    non_linear_pnp,
)
from src.bundle_adjustment import bundle_adjustment
from src.build_visibility_matrix import get_observations_index_and_viz_mat
from utils.data_utils import load_images, load_calibration
from utils.visualization_utils import (
    visualize_feature_matches,
    visualize_camera_poses,
    visualize_point_cloud,
    visualize_epipolar_lines,
    visualize_reprojection_errors,
)
from utils.logging_utils import setup_logging

import logging

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Structure from Motion Pipeline")

    parser = argparse.ArgumentParser(description='Structure from Motion Pipeline')
    parser.add_argument('--data_path', type=str, default='data/', help='Path to data')
    parser.add_argument('--output_path', type=str, default='outputs/', help='Path to save outputs')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to process')
    args = parser.parse_args()

    data_path = args.data_path
    output_path = args.output_path
    num_images = args.num_images

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Load images
    images_path = os.path.join(data_path, 'images')
    images = load_images(images_path, num_images)
    if not images:
        logger.error("No images found. Exiting.")
        return
    
    # Load calibration matrix K
    K = load_calibration(os.path.join(data_path, 'calibration.txt'))
    if K is None:
        logger.error("Calibration matrix K not found. Exiting.")
        return
    
    logger.info(f"Loaded calibration matrix K:\n{K}")

    # Feature extraction and matching
    feature_x, feature_y, feature_flag, feature_rgb_values = extract_features(
        data_path, num_images, logger
    )

    # Estimate fundamental matrices using RANSAC
    filtered_feature_flag, f_matrices = estimate_fundamental_matrices(
        feature_x, feature_y, feature_flag, num_images, output_path, images, logger
    )

    # Visualize feature matches and epipolar geometry for each image pair
    for i in range(num_images - 1):
        for j in range(i + 1, num_images):
            if f_matrices[i, j] is not None:
                idx = np.where(filtered_feature_flag[:, i] & filtered_feature_flag[:, j])
                pts1 = np.hstack(
                    (feature_x[idx, i].reshape(-1, 1), feature_y[idx, i].reshape(-1, 1))
                )
                pts2 = np.hstack(
                    (feature_x[idx, j].reshape(-1, 1), feature_y[idx, j].reshape(-1, 1))
                )
                visualize_feature_matches(images[i], images[j], pts1, pts2, output_path, i, j, logger)
                visualize_epipolar_lines(images[i], images[j], pts1, pts2, f_matrices[i, j], output_path, i, j, logger)

    # Process initial image pair
    idx = np.where(filtered_feature_flag[:, 0] & filtered_feature_flag[:, 1])
    pts1 = np.hstack(
        (
            feature_x[idx, 0].reshape(-1, 1),
            feature_y[idx, 0].reshape(-1, 1),
        )
    )
    pts2 = np.hstack(
        (
            feature_x[idx, 1].reshape(-1, 1),
            feature_y[idx, 1].reshape(-1, 1),
        )
    )

    # Compute Essential Matrix
    F = f_matrices[0, 1]
    E = compute_essential_matrix(K, F)

    # Extract and disambiguate camera poses
    R_set, C_set = extract_camera_poses(E)
    X_sets = []
    for i in range(len(R_set)):
        X = linear_triangulation(K, np.zeros((3, 1)), np.identity(3), C_set[i], R_set[i], pts1, pts2)
        X_sets.append(X)

    R, C, X = disambiguate_camera_pose(R_set, C_set, X_sets)

    # Non-linear triangulation
    X_refined = non_linear_triangulation(K, pts1, pts2, X, np.identity(3), np.zeros((3, 1)), R, C)

    # Initialize variables for bundle adjustment
    num_points = feature_x.shape[0]
    X_all = np.zeros((num_points, 3))
    X_found = np.zeros((num_points, 1), dtype=int)
    cam_indices = np.zeros((num_points, 1), dtype=int)

    idx_array = np.array(idx).reshape(-1)
    X_all[idx_array, :] = X_refined[:, :3]
    X_found[idx_array] = 1
    cam_indices[idx_array] = 1

    # Store initial camera poses
    C_set_global = [np.zeros((3, 1)), C.reshape(3, 1)]
    R_set_global = [np.identity(3), R]

    # Visualize initial point cloud
    visualize_point_cloud(X_refined[:, :3], output_path, step=0, logger=logger)

    # Process remaining images
    for i in range(2, num_images):
        logger.info(f"Processing image {i+1}")
        # Find common features between existing 3D points and new image
        idx_common = np.where(X_found[:, 0] & filtered_feature_flag[:, i])
        if len(idx_common[0]) < 8:
            logger.warning(f"Not enough common points with image {i+1}")
            continue

        pts_2d = np.hstack(
            (
                feature_x[idx_common, i].reshape(-1, 1),
                feature_y[idx_common, i].reshape(-1, 1),
            )
        )
        pts_3d = X_all[idx_common[0], :]

        # Estimate camera pose using PnP RANSAC
        R_init, C_init = pnp_ransac(K, pts_2d, pts_3d, logger=logger)
        if R_init is None or C_init is None:
            logger.error(f"PnP RANSAC failed for image {i+1}")
            continue

        # Non-linear PnP optimization
        R_opt, C_opt = non_linear_pnp(K, pts_2d, pts_3d, R_init, C_init)

        # Append to global camera poses
        R_set_global.append(R_opt)
        C_set_global.append(C_opt.reshape(3, 1))

        # Triangulate points between the new image and all previously registered images
        for j in range(i):
            idx_triangulate = np.where(filtered_feature_flag[:, j] & filtered_feature_flag[:, i])
            if len(idx_triangulate[0]) < 8:
                continue

            pts1 = np.hstack(
                (
                    feature_x[idx_triangulate, j].reshape(-1, 1),
                    feature_y[idx_triangulate, j].reshape(-1, 1),
                )
            )
            pts2 = np.hstack(
                (
                    feature_x[idx_triangulate, i].reshape(-1, 1),
                    feature_y[idx_triangulate, i].reshape(-1, 1),
                )
            )

            X_triangulated = linear_triangulation(
                K, C_set_global[j], R_set_global[j], C_opt.reshape(3, 1), R_opt, pts1, pts2
            )

            X_triangulated = X_triangulated / X_triangulated[:, 3].reshape(-1, 1)

            # Non-linear triangulation
            X_refined = non_linear_triangulation(
                K, pts1, pts2, X_triangulated, R_set_global[j], C_set_global[j], R_opt, C_opt
            )

            X_all[idx_triangulate[0], :] = X_refined[:, :3]
            X_found[idx_triangulate[0]] = 1

        # Visualize point cloud after adding new image
        visualize_point_cloud(X_all[X_found[:, 0] == 1], output_path, step=i, logger=logger)

        # Reprojection error visualization for new points
        pts_reproj = X_all[idx_triangulate[0], :]
        visualize_reprojection_errors(
            images[i],
            pts_2d,
            pts_3d,
            K,
            R_opt,
            C_opt,
            output_path,
            i,
            logger
        )

        # Bundle Adjustment
        logger.info("Starting Bundle Adjustment")
        X_index, visibility_matrix = get_observations_index_and_viz_mat(
            X_found, filtered_feature_flag, nCam=i
        )
        R_set_global, C_set_global, X_all = bundle_adjustment(
            X_index,
            visibility_matrix,
            X_all,
            X_found,
            feature_x,
            feature_y,
            filtered_feature_flag,
            R_set_global,
            C_set_global,
            K,
            nCam=i,
            logger=logger,
        )

    # Final visualization
    visualize_point_cloud(X_all[X_found[:, 0] == 1], output_path, step='final', logger=logger)
    visualize_camera_poses(C_set_global, R_set_global, X_all[X_found[:, 0] == 1], output_path, logger=logger)

    logger.info("Structure from Motion Pipeline Completed")

if __name__ == '__main__':
    main()
