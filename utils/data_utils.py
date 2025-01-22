import os
import cv2
import numpy as np
import logging

def load_images(images_path, num_images):
    logger = logging.getLogger(__name__)
    images = []
    for i in range(1, num_images + 1):
        image_path = os.path.join(images_path, f"{i}.png")
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)
        else:
            logger.error(f"No image found at {image_path}")
    return images

def load_calibration(calibration_file):
    logger = logging.getLogger(__name__)
    if os.path.isfile(calibration_file):
        try:
            K = np.loadtxt(calibration_file)
            return K
        except Exception as e:
            logger.error(f"Error reading calibration file {calibration_file}: {e}")
            return None
    else:
        logger.error(f"Calibration file {calibration_file} not found.")
        return None
