import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging

def visualize_feature_matches(img1, img2, pts1, pts2, output_path, img1_idx, img2_idx, logger=None):
    if logger:
        logger.info(f"Visualizing feature matches between images {img1_idx+1} and {img2_idx+1}")

    # Draw matches
    img_matches = cv2.hconcat([img1, img2])
    for pt1, pt2 in zip(pts1, pts2):
        pt1 = tuple(map(int, pt1))
        pt2 = (int(pt2[0] + img1.shape[1]), int(pt2[1]))
        cv2.circle(img_matches, pt1, 5, (0, 255, 0), -1)
        cv2.circle(img_matches, pt2, 5, (0, 255, 0), -1)
        cv2.line(img_matches, pt1, pt2, (255, 0, 0), 1)

    output_file = os.path.join(output_path, f"feature_matches_{img1_idx+1}_{img2_idx+1}.png")
    cv2.imwrite(output_file, img_matches)
    if logger:
        logger.info(f"Saved feature matches visualization: {output_file}")

def visualize_epipolar_lines(img1, img2, pts1, pts2, F, output_path, img1_idx, img2_idx, logger=None):
    if logger:
        logger.info(f"Visualizing epipolar lines between images {img1_idx+1} and {img2_idx+1}")

    def draw_epipolar_lines(img, lines, pts):
        _, c = img.shape[:2]
        img_epilines = img.copy()
        for line, pt in zip(lines, pts):
            x0, y0 = map(int, [0, -line[2] / line[1]])
            x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
            pt = tuple(map(int, pt))
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(img_epilines, (x0, y0), (x1, y1), color, 1)
            cv2.circle(img_epilines, pt, 5, color, -1)
        return img_epilines

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    img1_epilines = draw_epipolar_lines(img1, lines1, pts1)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    img2_epilines = draw_epipolar_lines(img2, lines2, pts2)

    combined_img = cv2.hconcat([img1_epilines, img2_epilines])
    output_file = os.path.join(output_path, f"epipolar_lines_{img1_idx+1}_{img2_idx+1}.png")
    cv2.imwrite(output_file, combined_img)
    if logger:
        logger.info(f"Saved epipolar lines visualization: {output_file}")

def visualize_camera_poses(C_set, R_set, X, output_path, logger=None):
    if logger:
        logger.info("Visualizing camera poses and 3D points")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='.', s=1)

    for i, (C, R) in enumerate(zip(C_set, R_set)):
        ax.scatter(C[0], C[1], C[2], marker='o', s=50, label=f"Camera {i+1}")
        # Draw orientation arrow based on rotation matrix
        arrow = R @ np.array([0.1, 0, 0])  # X-axis direction in camera frame
        ax.quiver(C[0], C[1], C[2], arrow[0], arrow[1], arrow[2], color="r")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    output_file = os.path.join(output_path, "camera_poses.png")
    plt.savefig(output_file)
    if logger:
        logger.info(f"Saved camera poses visualization: {output_file}")

def visualize_point_cloud(X, output_path, step="final", logger=None):
    if logger:
        logger.info(f"Visualizing 3D point cloud at {step} step")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='.', s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    output_file = os.path.join(output_path, f"point_cloud_{step}.png")
    plt.savefig(output_file)
    if logger:
        logger.info(f"Saved 3D point cloud visualization: {output_file}")


def visualize_reprojection_errors(img, pts2D, pts3D, K, R, C, output_path, img_idx, logger=None):
    if logger:
        logger.info(f"Visualizing reprojection errors for image {img_idx+1}")

    # Ensure C is a column vector
    C = C.reshape(3, 1)

    # Compute the projection matrix P = K * [R | -R * C]
    P = K @ R @ np.hstack((np.identity(3), -C))

    # Convert pts3D to homogeneous coordinates (N x 4)
    pts3D_homogeneous = np.hstack((pts3D, np.ones((pts3D.shape[0], 1))))

    # Project 3D points to 2D (resulting shape: N x 3)
    pts_reproj_homogeneous = (P @ pts3D_homogeneous.T).T

    # Normalize to get 2D points
    pts_reproj_2D = pts_reproj_homogeneous[:, :2] / pts_reproj_homogeneous[:, 2, np.newaxis]

    img_errors = img.copy()
    for pt2D, pt_reproj in zip(pts2D, pts_reproj_2D):
        pt2D = tuple(map(int, pt2D))
        pt_reproj = tuple(map(int, pt_reproj))
        cv2.circle(img_errors, pt2D, 5, (0, 255, 0), -1)  # Original point in green
        cv2.circle(img_errors, pt_reproj, 5, (0, 0, 255), -1)  # Reprojected point in red
        cv2.line(img_errors, pt2D, pt_reproj, (255, 0, 0), 1)

    output_file = os.path.join(output_path, f"reprojection_errors_{img_idx+1}.png")
    cv2.imwrite(output_file, img_errors)
    if logger:
        logger.info(f"Saved reprojection error visualization: {output_file}")