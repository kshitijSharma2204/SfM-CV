# src/feature_extraction.py

import os
import numpy as np
import logging

def extract_features(data_path, num_images, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    feature_rgb_values = []
    feature_x = []
    feature_y = []
    feature_flag = []

    matching_files_path = os.path.join(data_path, 'matching_files')

    for n in range(1, num_images):
        matching_file_name = f"matching{n}.txt"
        file = os.path.join(matching_files_path, matching_file_name)
        if not os.path.isfile(file):
            logger.error(f"Matching file {file} not found.")
            continue

        with open(file, "r") as matching_file:
            for i, row in enumerate(matching_file):
                if i == 0:
                    nfeatures = int(row.split(':')[1])
                else:
                    x_row = np.zeros((1, num_images))
                    y_row = np.zeros((1, num_images))
                    flag_row = np.zeros((1, num_images), dtype=int)
                    columns = np.array([float(x) for x in row.strip().split()])

                    nMatches = int(columns[0])
                    r_value, b_value, g_value = columns[1:4]
                    feature_rgb_values.append([r_value, g_value, b_value])
                    current_x, current_y = columns[4:6]

                    x_row[0, n - 1] = current_x
                    y_row[0, n - 1] = current_y
                    flag_row[0, n - 1] = 1

                    m = 0
                    while nMatches > 1:
                        idx = 6 + m * 3
                        if idx + 2 >= len(columns):
                            break
                        image_id = int(columns[idx])
                        image_id_x = columns[idx + 1]
                        image_id_y = columns[idx + 2]
                        m += 1
                        nMatches -= 1

                        # Adjust indexing for image_id
                        if 1 <= image_id <= num_images:
                            x_row[0, image_id - 1] = image_id_x
                            y_row[0, image_id - 1] = image_id_y
                            flag_row[0, image_id - 1] = 1
                        else:
                            logger.warning(f"Invalid image_id {image_id} in {file}")

                    feature_x.append(x_row)
                    feature_y.append(y_row)
                    feature_flag.append(flag_row)

    feature_x = np.asarray(feature_x).reshape(-1, num_images)
    feature_y = np.asarray(feature_y).reshape(-1, num_images)
    feature_flag = np.asarray(feature_flag).reshape(-1, num_images)
    feature_rgb_values = np.asarray(feature_rgb_values).reshape(-1, 3)
    logger.info(f"Extracted features from matching files")
    return feature_x, feature_y, feature_flag, feature_rgb_values