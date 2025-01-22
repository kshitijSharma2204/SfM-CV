import numpy as np
from src.estimation_utils import (
    get_inliers_ransac,
)
from utils.visualization_utils import visualize_feature_matches, visualize_epipolar_lines
import logging

def estimate_fundamental_matrices(
    feature_x, feature_y, feature_flag, num_images, output_path, images, logger=None
):
    if logger is None:
        logger = logging.getLogger(__name__)

    filtered_feature_flag = np.zeros_like(feature_flag)
    f_matrices = np.empty((num_images, num_images), dtype=object)

    for i in range(num_images - 1):
        for j in range(i + 1, num_images):
            idx = np.where(feature_flag[:, i] & feature_flag[:, j])
            pts1 = np.hstack(
                (
                    feature_x[idx, i].reshape(-1, 1),
                    feature_y[idx, i].reshape(-1, 1),
                )
            )
            pts2 = np.hstack(
                (
                    feature_x[idx, j].reshape(-1, 1),
                    feature_y[idx, j].reshape(-1, 1),
                )
            )
            idx = np.array(idx).reshape(-1)

            if len(idx) > 8:
                F_inliers, inliers_idx = get_inliers_ransac(pts1, pts2, idx, logger=logger)
                logger.info(
                    f"Between Images {i+1} and {j+1}, Inliers: {len(inliers_idx)} / {len(idx)}"
                )
                f_matrices[i, j] = F_inliers
                filtered_feature_flag[idx[inliers_idx], j] = 1
                filtered_feature_flag[idx[inliers_idx], i] = 1

                # Visualize feature matches
                visualize_feature_matches(
                    images[i],
                    images[j],
                    pts1[inliers_idx],
                    pts2[inliers_idx],
                    output_path,
                    i,
                    j,
                    logger=logger,
                )

                # Visualize epipolar lines
                visualize_feature_matches(
                    images[i],
                    images[j],
                    pts1[inliers_idx],
                    pts2[inliers_idx],
                    output_path,
                    i,
                    j,
                    logger=logger,
                )
            else:
                logger.warning(f"Not enough points between images {i+1} and {j+1}")

    return filtered_feature_flag, f_matrices