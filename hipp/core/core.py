import concurrent
import concurrent.futures
import os
import pathlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import rasterio
from skimage import transform as tf
from tqdm import tqdm
from tqdm.auto import tqdm

import hipp.core
import hipp.image
import hipp.io
import hipp.math
import hipp.plot
import hipp.qc
import hipp.tools
import hipp.utils

"""
Library with core image pre-processing functions.
"""


def compute_principal_point(
    subpixel_fiducial_locations, subpixel_quality_scores, median_scores, threshold=0.01
):

    principal_point_estimates = []

    A0, B0, A1, B1 = subpixel_fiducial_locations
    A0_score, B0_score, A1_score, B1_score = subpixel_quality_scores
    A0_median_score, B0_median_score, A1_median_score, B1_median_score = median_scores

    principal_point_A = hipp.core.eval_and_compute_principal_point(
        A0, A0_score, A0_median_score, A1, A1_score, A1_median_score, threshold
    )
    if principal_point_A:
        principal_point_estimates.append(principal_point_A)

    principal_point_B = hipp.core.eval_and_compute_principal_point(
        B0, B0_score, B0_median_score, B1, B1_score, B1_median_score, threshold
    )
    if principal_point_B:
        principal_point_estimates.append(principal_point_B)

    principal_point_estimates = np.array(principal_point_estimates)
    if principal_point_estimates.size != 0:
        principal_point = (
            principal_point_estimates[:, 0].mean(),
            principal_point_estimates[:, 1].mean(),
        )
        return principal_point


def compute_principal_points(fiducial_locations_df, quality_scores_df, threshold=0.01):

    median_scores = []
    for i in np.arange(0, 4):
        median_score = quality_scores_df.iloc[:, i].median()
        median_scores.append(median_score)

    principal_points = []
    for i in range(len(quality_scores_df)):
        subpixel_fiducial_locations = fiducial_locations_df.iloc[i].values
        subpixel_quality_scores = quality_scores_df.iloc[i].values

        principal_point = hipp.core.compute_principal_point(
            subpixel_fiducial_locations,
            subpixel_quality_scores,
            median_scores,
            threshold,
        )
        principal_points.append([principal_point])

    principal_points_df = pd.DataFrame(principal_points, columns=["principal_point"])

    return principal_points_df


def compute_mean_midside_corner_principal_point(df_corner, df_midside):
    df = pd.concat(
        [
            pd.DataFrame(
                df_midside.principal_point.to_list(), columns=["midside_y", "midside_x"]
            ),
            pd.DataFrame(
                df_corner.principal_point.to_list(), columns=["corner_y", "corner_x"]
            ),
        ],
        axis=1,
    )
    df["principal_point"] = list(
        zip(
            df[["midside_y", "corner_y"]].mean(axis=1),
            df[["midside_x", "corner_x"]].mean(axis=1),
        )
    )

    return df


def compute_principal_point_from_proxies(df, verbose=True):
    distances = []
    principal_points = []
    intersection_angles = []
    verbose = False

    for index, row in df.iterrows():
        if verbose:
            print("Computing principal point for:", row["file_names"])
        p1 = (row["left_y"], row["left_x"])
        p2 = (row["right_y"], row["right_x"])
        principal_point_LR = hipp.math.midpoint(p1[1], p1[0], p2[1], p2[0])
        distances.append(hipp.math.distance(p1, p2))

        p1 = (row["top_y"], row["top_x"])
        p2 = (row["bottom_y"], row["bottom_x"])
        principal_point_TB = hipp.math.midpoint(p1[1], p1[0], p2[1], p2[0])
        distances.append(hipp.math.distance(p1, p2))

        # if no diametrically opposing proxies are found
        # use first viable combination of left/right x or top/bottom y
        # to estimate position
        if np.isnan(principal_point_LR).any() and np.isnan(principal_point_TB).any():
            if np.isnan(principal_point_LR).any():
                principal_point_LR = (row["left_y"], row["top_x"])
            if np.isnan(principal_point_LR).any():
                principal_point_LR = (row["left_y"], row["bottom_x"])
            if np.isnan(principal_point_LR).any():
                principal_point_LR = (row["right_y"], row["top_x"])
            if np.isnan(principal_point_LR).any():
                principal_point_LR = (row["right_y"], row["bottom_x"])

            if np.isnan(principal_point_LR).any():
                if np.isnan(principal_point_TB).any():
                    principal_point_TB = (row["left_y"], row["top_x"])
                if np.isnan(principal_point_TB).any():
                    principal_point_TB = (row["left_y"], row["bottom_x"])
                if np.isnan(principal_point_TB).any():
                    principal_point_TB = (row["right_y"], row["top_x"])
                if np.isnan(principal_point_TB).any():
                    principal_point_TB = (row["right_y"], row["bottom_x"])

        if np.isnan(principal_point_LR).any() and np.isnan(principal_point_TB).any():
            #             if verbose:
            print("WARNING: Unable to estimate principal point for:", row["file_names"])
            print(
                "WARNING: Using mean principal point estimate from image set instead."
            )
            principal_point = (np.nan, np.nan)
            principal_points.append(principal_point)
        else:
            principal_point = tuple(
                map(np.nanmean, zip(*(principal_point_TB, principal_point_LR)))
            )
            principal_point = np.array([int(round(x)) for x in principal_point])
            principal_points.append(principal_point)
            if verbose:
                print("Principal point estimated at:", str(principal_point))

        proxy_locations = np.array(list(zip(row.values[1::2], row.values[2::2])))
        intersection_angle = hipp.qc.compute_opposing_fiducial_intersection_angle(
            proxy_locations
        )
        intersection_angles.append(intersection_angle)
        if verbose:
            if not np.isnan(intersection_angle) and verbose:
                print("Intersection angle at principal point:", str(intersection_angle))
            elif verbose:
                print(
                    "Insufficient fiducial proxies (<4) detected to compute intersection angle."
                )

    # Use mean principal point estimate from image set to replace instance where < 2 proxies were found.
    df_tmp = pd.DataFrame(principal_points)
    principal_points = list(
        df_tmp.fillna(df_tmp.mean().round().astype(int)).astype(int).values
    )

    return principal_points, distances, intersection_angles


def validate_square_dim(
    image_files, buffer_distance, principal_points, image_square_dim
):
    new_square_dims = []

    for i, v in enumerate(image_files):
        ds = rasterio.open(v)
        h = ds.height + buffer_distance * 2
        w = ds.width + buffer_distance * 2
        pp_h = principal_points[i][0] + buffer_distance / 2
        pp_w = principal_points[i][1] + buffer_distance / 2

        tmp_h = pp_h + image_square_dim / 2
        tmp_w = pp_w + image_square_dim / 2

        if tmp_w > w:
            new_square_dims.append(image_square_dim - (tmp_w - w))
        if tmp_h > h:
            new_square_dims.append(image_square_dim - (tmp_h - h))

    if new_square_dims:
        new_square_dim = int(np.floor(np.nanmin(new_square_dims)))
        return new_square_dim
    else:
        return None


def create_fiducial_template(
    image_file,
    df=None,
    output_directory="fiducials",
    output_file_name="fiducial.tif",
    distance_around_fiducial=100,
):

    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)

    image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    if isinstance(df, type(None)):
        df = hipp.tools.point_picker(image_file)

    fiducial = (df.x[0], df.y[0])

    x_L = round(fiducial[0] - distance_around_fiducial)
    x_R = round(fiducial[0] + distance_around_fiducial)
    y_T = round(fiducial[1] - distance_around_fiducial)
    y_B = round(fiducial[1] + distance_around_fiducial)
    cropped = image_array[y_T:y_B, x_L:x_R]

    out = os.path.join(output_directory, output_file_name)
    cv2.imwrite(out, cropped)

    return out


def create_midside_fiducial_proxies_template(
    image_file,
    df=None,
    output_directory="input_data/fiducials",
    buffer_distance=250,
    threshold=50,
):

    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)

    image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    n, bins, patches = plt.hist(image_array.ravel()[::40], bins=256, range=(0, 256))
    #     plt.close()

    #     p = find_peaks(n,prominence=1, width=1, height=n.max()/3)
    #     p = find_peaks(n,prominence=100,width=1)

    #     threshold = p[1]['right_bases'][0]
    #     print(threshold)

    plt.vlines(threshold, 0, n.max(), "r")
    image_array = hipp.image.threshold_and_add_noise(image_array, threshold=threshold)
    image_array = hipp.image.clahe_equalize_image(image_array)
    image_array = hipp.image.img_linear_stretch(image_array)

    image_array = hipp.core.pad_image(image_array, buffer_distance=buffer_distance)
    if isinstance(df, type(None)):
        print(
            "Select inner most point to crop from for midside fiducial marker proxies,"
        )
        print("in order from Left - Top - Right - Bottom.")
        df = hipp.tools.point_picker(image_file, point_count=4)

    df = df + buffer_distance

    left_fiducial = (df.x[0], df.y[0])
    top_fiducial = (df.x[1], df.y[1])
    right_fiducial = (df.x[2], df.y[2])
    bottom_fiducial = (df.x[3], df.y[3])

    fiducials = [left_fiducial, top_fiducial, right_fiducial, bottom_fiducial]

    dist_w, dist_h = buffer_distance, buffer_distance

    x_L = int(left_fiducial[0] - dist_w)
    x_R = int(left_fiducial[0])
    y_T = int(left_fiducial[1] - 2 * dist_w)
    y_B = int(left_fiducial[1] + 2 * dist_w)
    cropped = image_array[y_T:y_B, x_L:x_R]
    cv2.imwrite(os.path.join(output_directory, "L.tif"), cropped)

    x_L = int(top_fiducial[0] - 2 * dist_h)
    x_R = int(top_fiducial[0] + 2 * dist_h)
    y_T = int(top_fiducial[1] - dist_h)
    y_B = int(top_fiducial[1])
    cropped = image_array[y_T:y_B, x_L:x_R]
    cv2.imwrite(os.path.join(output_directory, "T.tif"), cropped)

    x_L = int(right_fiducial[0])
    x_R = int(right_fiducial[0] + dist_w)
    y_T = int(right_fiducial[1] - 2 * dist_w)
    y_B = int(right_fiducial[1] + 2 * dist_w)
    cropped = image_array[y_T:y_B, x_L:x_R]
    cv2.imwrite(os.path.join(output_directory, "R.tif"), cropped)

    x_L = int(bottom_fiducial[0] - 2 * dist_h)
    x_R = int(bottom_fiducial[0] + 2 * dist_h)
    y_T = int(bottom_fiducial[1])
    y_B = int(bottom_fiducial[1] + dist_h)
    cropped = image_array[y_T:y_B, x_L:x_R]
    cv2.imwrite(os.path.join(output_directory, "B.tif"), cropped)

    return output_directory


def crop_fiducial(
    image_file,
    image_array,
    match,
    label=None,
    distance_from_loc=200,
    output_directory="tmp/fiducial_crop",
):

    x_L = match[1]
    x_R = match[1] + distance_from_loc
    y_T = match[0]
    y_B = match[0] + distance_from_loc
    fiducial_crop_array = image_array[y_T:y_B, x_L:x_R]

    file_path, file_name, file_extension = hipp.io.split_file(image_file)

    output_file_name = os.path.join(
        output_directory, file_name + "_" + label + file_extension
    )

    cv2.imwrite(output_file_name, fiducial_crop_array)

    return output_file_name


def crop_image_from_file(
    image_file_principal_point_tuple,
    image_square_dim,
    output_directory="input_data/cropped_images",
    buffer_distance=250,
    stretch_histogram=True,
    clahe_enhancement=True,
):

    image_file, principal_point = image_file_principal_point_tuple

    image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    image_array = hipp.core.pad_image(image_array, buffer_distance=buffer_distance)

    image_array = hipp.image.crop_about_point(
        image_array, principal_point, image_square_dim=image_square_dim
    )

    if clahe_enhancement:
        image_array = hipp.image.clahe_equalize_image(image_array)
    if stretch_histogram:
        image_array = hipp.image.img_linear_stretch(image_array)

    path, basename, extension = hipp.io.split_file(image_file)
    out = os.path.join(output_directory, basename + extension)
    cv2.imwrite(out, image_array)
    return out


def define_midside_windows(
    image_array,
    reduce_left_window_by_fraction=0,
    reduce_top_window_by_fraction=0,
    reduce_right_window_by_fraction=0,
    reduce_bottom_window_by_fraction=0,
):

    half_image_height = int(image_array.shape[0] / 2)
    quarter_image_height = int(half_image_height / 2)

    half_image_width = int(image_array.shape[1] / 2)
    quarter_image_width = int(half_image_width / 2)

    midside_left = [
        int(
            quarter_image_height + quarter_image_height * reduce_left_window_by_fraction
        ),
        int(
            half_image_height
            + quarter_image_height
            - quarter_image_height * reduce_left_window_by_fraction
        ),
        0,
        int(quarter_image_width - quarter_image_width * reduce_left_window_by_fraction),
    ]

    midside_top = [
        0,
        int(
            quarter_image_height - quarter_image_height * reduce_top_window_by_fraction
        ),
        int(quarter_image_width + quarter_image_width * reduce_top_window_by_fraction),
        int(
            half_image_width
            + quarter_image_width
            - quarter_image_width * reduce_top_window_by_fraction
        ),
    ]

    midside_right = [
        int(
            quarter_image_height
            + quarter_image_height * reduce_right_window_by_fraction
        ),
        int(
            half_image_height
            + quarter_image_height
            - quarter_image_height * reduce_right_window_by_fraction
        ),
        half_image_width
        + quarter_image_width
        + int(quarter_image_width * reduce_right_window_by_fraction),
        image_array.shape[1],
    ]

    midside_bottom = [
        half_image_height
        + quarter_image_height
        + int(quarter_image_height * reduce_bottom_window_by_fraction),
        image_array.shape[0],
        int(
            quarter_image_width + quarter_image_width * reduce_bottom_window_by_fraction
        ),
        int(
            half_image_width
            + quarter_image_width
            - +quarter_image_width * reduce_bottom_window_by_fraction
        ),
    ]

    #     midside_left = [5900, 6500,0, 1500]
    #     midside_top = [0, 500,6300, 7700]
    #     midside_right = [5900, 6500, 12800, 13251]
    #     midside_bottom = [11900, 12432,6400, 7700]

    midside_windows = [midside_left, midside_top, midside_right, midside_bottom]

    return midside_windows


def define_corner_windows(image_array):

    half_image_height = int(image_array.shape[0] / 2)
    quarter_image_height = int(half_image_height / 2)

    half_image_width = int(image_array.shape[1] / 2)
    quarter_image_width = int(half_image_width / 2)

    corner_top_left = [0, quarter_image_height, 0, quarter_image_width]

    corner_top_right = [
        0,
        quarter_image_height,
        half_image_width + quarter_image_width,
        image_array.shape[1],
    ]

    corner_bottom_right = [
        half_image_height + quarter_image_height,
        image_array.shape[0],
        half_image_width + quarter_image_width,
        image_array.shape[1],
    ]

    corner_bottom_left = [
        half_image_height + quarter_image_height,
        image_array.shape[0],
        0,
        quarter_image_width,
    ]

    corner_windows = [
        corner_top_left,
        corner_top_right,
        corner_bottom_right,
        corner_bottom_left,
    ]

    return corner_windows


def define_center_window(image_array):

    # define window as slices: [y1:y2, x1:x2], so window = [y1, y2, x1, x2]

    half_image_height = int(image_array.shape[0] / 2)
    quarter_image_height = int(half_image_height / 2)

    half_image_width = int(image_array.shape[1] / 2)
    quarter_image_width = int(half_image_width / 2)

    center_window = [
        [
            int(np.round(half_image_height - quarter_image_height / 2)),
            int(np.round(half_image_height + quarter_image_height / 2)),
            int(np.round(half_image_width - quarter_image_width / 2)),
            int(np.round(half_image_width + quarter_image_width / 2)),
        ]
    ]

    return center_window


def detect_fiducials(slices, template_array, windows):

    matches = []
    quality_scores = []

    for index, slice_array in enumerate(slices):
        match_location, quality_score = hipp.core.match_template(
            slice_array, template_array
        )

        match = (
            windows[index][0] + match_location[0],
            windows[index][2] + match_location[1],
        )
        matches.append(match)
        quality_scores.append(quality_score)

    return matches, quality_scores


def detect_fiducial_proxies(image_file, templates, buffer_distance=250):

    image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    #     image_array = cv2.imread(image_file,cv2.IMREAD_COLOR)
    #     image_array = image_array[:,:,0]

    #     n, bins, patches = plt.hist(image_array.ravel()[::40],bins=256,range=(0,256))
    #     plt.close()
    #     p = find_peaks(n,prominence=10, width=1, height=n.max()/3)
    #     threshold = p[1]['right_bases'][0]
    #     image_array = hipp.image.threshold_and_add_noise(image_array, threshold=threshold)

    image_array = hipp.image.clahe_equalize_image(image_array)
    image_array = hipp.image.img_linear_stretch(image_array)
    #     image_array = hipp.image.threshold_and_add_noise(image_array)

    image_array = hipp.core.pad_image(image_array, buffer_distance=buffer_distance)
    windows = hipp.core.define_midside_windows(image_array)
    slices = hipp.core.slice_image_frame(image_array, windows)

    matches = []
    quality_scores = []

    for index, slice_array in enumerate(slices):
        template = templates[index]

        #         n, bins, patches = plt.hist(template.ravel()[::40],bins=256,range=(0,256))
        #         plt.close()
        #         p = find_peaks(n,prominence=10, width=1, height=n.max()/3)
        #         threshold = p[1]['right_bases'][0]
        #         template = hipp.image.threshold_and_add_noise(template.copy(), threshold=threshold)

        #         template = hipp.image.clahe_equalize_image(template.copy())
        #         template = hipp.image.img_linear_stretch(template.copy())
        #         template = hipp.image.threshold_and_add_noise(template.copy())

        match_location, quality_score = hipp.core.match_template(slice_array, template)
        match = (
            windows[index][0] + match_location[0],
            windows[index][2] + match_location[1],
        )
        matches.append(match)
        quality_scores.append(quality_score)

    left, top, right, bottom = matches

    left_t_shape = templates[0].shape
    top_t_shape = templates[1].shape
    right_t_shape = templates[2].shape
    bottom_t_shape = templates[3].shape

    left_fiducial = (left[0] + left_t_shape[0] / 2, left[1] + left_t_shape[1])
    top_fiducial = (top[0] + top_t_shape[0], top[1] + top_t_shape[1] / 2)
    right_fiducial = (right[0] + right_t_shape[0] / 2, right[1])
    bottom_fiducial = (bottom[0], bottom[1] + bottom_t_shape[1] / 2)

    matches = [left_fiducial, top_fiducial, right_fiducial, bottom_fiducial]

    return matches, quality_scores, image_file


def detect_high_res_fiducial(
    fiducial_crop_high_res_file,
    template_high_res_zoomed_file,
    distance_from_loc=200,
    qc=True,
    qc_directory="qc/fiducial_detection",
):

    fiducial_crop_high_res_array = cv2.imread(
        fiducial_crop_high_res_file, cv2.IMREAD_GRAYSCALE
    )
    template_high_res_zoomed_array = cv2.imread(
        template_high_res_zoomed_file, cv2.IMREAD_GRAYSCALE
    )

    if fiducial_crop_high_res_array is None or template_high_res_zoomed_array is None:
        raise ValueError("Failed to load image files")

    match_location, quality_score = hipp.core.match_template(
        fiducial_crop_high_res_array, template_high_res_zoomed_array
    )

    if qc:
        output_directory = qc_directory
        p = pathlib.Path(output_directory)
        p.mkdir(parents=True, exist_ok=True)

        file_path, file_name, file_extension = hipp.io.split_file(
            fiducial_crop_high_res_file
        )

        image_array = cv2.cvtColor(fiducial_crop_high_res_array, cv2.COLOR_GRAY2RGB)
        image_array[
            (
                match_location[0] + int(distance_from_loc / 2),
                match_location[1] + int(distance_from_loc / 2),
            )
        ] = 255, 0, 0

        x_L = match_location[1]
        x_R = match_location[1] + distance_from_loc
        y_T = match_location[0]
        y_B = match_location[0] + distance_from_loc
        image_array = image_array[y_T:y_B, x_L:x_R]

        out = os.path.join(output_directory, file_name + file_extension)
        cv2.imwrite(out, image_array)

    return match_location, quality_score


def detect_subpixel_fiducial_coordinates(
    image_file,
    image_array,
    matches,
    template_high_res_zoomed_file,
    labels: list | None = None,
    distance_from_loc=200,
    factor=8,
    qc=True,
    qc_directory="qc/fiducial_detection",
):

    import tempfile

    if labels is None:
        labels = ["midside_left", "midside_top", "midside_right", "midside_bottom"]

    subpixel_fiducial_locations = []
    quality_scores = []

    # Create a unique temporary directory for this worker process
    # Automatically cleaned up when exiting the context
    with tempfile.TemporaryDirectory(prefix="fiducial_crop_") as output_directory:
        for index, match_location in enumerate(matches):
            cropped_fiducial_file = hipp.core.crop_fiducial(
                image_file,
                image_array,
                match_location,
                label=labels[index],
                distance_from_loc=distance_from_loc,
                output_directory=output_directory,
            )

            fiducial_crop_high_res_file = hipp.utils.enhance_geotif_resolution(
                cropped_fiducial_file, factor=factor
            )

            match_location_high_res, quality_score = hipp.core.detect_high_res_fiducial(
                fiducial_crop_high_res_file,
                template_high_res_zoomed_file,
                distance_from_loc=distance_from_loc,
                qc=qc,
                qc_directory=qc_directory,
            )

            template_hr = cv2.imread(
                template_high_res_zoomed_file, cv2.IMREAD_GRAYSCALE
            )
            half_h = template_hr.shape[0] // 2
            half_w = template_hr.shape[1] // 2
            y, x = (
                (match_location_high_res[0] + half_h) / factor,
                (match_location_high_res[1] + half_w) / factor,
            )
            subpixel_fiducial_location = y + match_location[0], x + match_location[1]

            subpixel_fiducial_locations.append(subpixel_fiducial_location)
            quality_scores.append(quality_score)

    # Cleanup happens automatically when exiting the 'with' block
    return subpixel_fiducial_locations, quality_scores


def eval_and_compute_principal_point(
    P1, P1_score, P1_median_score, P2, P2_score, P2_median_score, threshold=0.01
):
    """
    Evaluates fiducial point detection score and computes principal point estimates as midpoint between diametrically opposed fiducial markers.
    """

    if (
        P1_median_score - P1_score < threshold
        and P2_median_score - P2_score < threshold
    ):
        principal_point = hipp.math.midpoint(P1[1], P1[0], P2[1], P2[0])
        return principal_point


def eval_matches(df, split_position_tuples=False, threshold=0.01):
    """
    Replaces fiducial marker positions that received a low score with np.nan in place.
    A low score is determined by the difference between the median score for a given fiducial marker position
    across all images and a given score exceeding the threshold.
    Removes score columns and splits position tuples into seperate columns.
    """
    df = hipp.core.nan_low_scoring_fiducial_matches(df, threshold=threshold)

    columns = df.columns.values
    columns = [x for x in columns if "score" not in x]
    df = df[columns]

    if split_position_tuples:
        df = hipp.core.split_position_tuples(df)

    return df


def iter_crop_image_from_file(
    images,
    principal_points,
    image_square_dim,
    output_directory="input_data/cropped_images",
    buffer_distance=250,
    stretch_histogram=True,
    clahe_enhancement=True,
    verbose=True,
    max_workers=None,
):

    print("Cropping images...")

    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)

    max_workers = max_workers or psutil.cpu_count(logical=True) - 1

    with tqdm(total=len(images)) as pbar:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        future = {
            pool.submit(
                hipp.core.crop_image_from_file,
                img_pp,
                image_square_dim,
                buffer_distance=buffer_distance,
                output_directory=output_directory,
                stretch_histogram=stretch_histogram,
                clahe_enhancement=clahe_enhancement,
            ): img_pp
            for img_pp in zip(images, principal_points)
        }
        results = []
        for f in concurrent.futures.as_completed(future):
            r = f.result()
            pbar.update(1)
    print("Cropped images at:", output_directory)


def iter_detect_fiducial_proxies(
    images, templates, buffer_distance=250, max_workers=None, verbose=False
):
    print("Detecting fiducial proxies...")

    if max_workers is None:
        max_workers = psutil.cpu_count(logical=True) - 1

    results = []
    with tqdm(total=len(images)) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future = {
                pool.submit(
                    hipp.core.detect_fiducial_proxies,
                    image_file,
                    templates,
                    buffer_distance=buffer_distance,
                ): image_file
                for image_file in images
            }
            for f in as_completed(future):
                results.append(f.result())
                pbar.update(1)
    df = (
        pd
        .DataFrame(results, columns=["match_locations", "scores", "file_names"])
        .sort_values(by=["file_names"])
        .reset_index(drop=True)
    )
    return df


def load_midside_fiducial_proxy_templates(template_directory):

    # need to get rid of old jpg templates and recreate for all marker types
    l_tif_path = os.path.join(template_directory, "L.tif")
    l_jpg_path = os.path.join(template_directory, "L.jpg")

    assert os.path.exists(l_tif_path) or os.path.exists(l_jpg_path), (
        f'Fiducial marker files "L.tif" or "L.jpg" must exist in provided directory {template_directory}.'
    )

    extension = ".tif" if os.path.exists(l_tif_path) else ".jpg"

    L = os.path.join(template_directory, "L" + extension)
    T = os.path.join(template_directory, "T" + extension)
    R = os.path.join(template_directory, "R" + extension)
    B = os.path.join(template_directory, "B" + extension)

    template_files = [L, T, R, B]
    templates = []

    for t in template_files:
        template = cv2.imread(t, cv2.IMREAD_GRAYSCALE)
        templates.append(template)

    return templates


def match_template(image_array, template_array):

    #     image_array = np.where(image_array>200,image_array,0)
    #     template_array = np.where(template_array>200,template_array,0)

    result = cv2.matchTemplate(image_array, template_array, cv2.TM_CCOEFF_NORMED)
    location = np.where(result == result.max())

    match_location = (location[0][0], location[1][0])
    quality_score = result.max()

    return match_location, quality_score


def merge_midside_df_corner_df(
    df_corner=None, df_midside=None, file_name_column="fileName"
):

    if isinstance(df_midside, Iterable) and isinstance(df_corner, Iterable):
        df = hipp.core.compute_mean_midside_corner_principal_point(
            df_corner, df_midside
        )

        del df_midside["principal_point"]
        del df_corner["principal_point"]

        df_detected = pd.merge(df_midside, df_corner, on=file_name_column)
        df_detected = pd.concat([df_detected, df["principal_point"]], axis=1)
        df_detected = hipp.core.split_position_tuples(df_detected)
        return df_detected

    elif isinstance(df_midside, Iterable) and not isinstance(df_corner, Iterable):
        df_midside = hipp.core.split_position_tuples(df_midside)
        return df_midside

    elif isinstance(df_corner, Iterable) and not isinstance(df_midside, Iterable):
        df_corner = hipp.core.split_position_tuples(df_corner)
        return df_corner


def nan_low_scoring_fiducial_matches(df, threshold=0.01):
    """
    Replaces fiducial marker positions that received a low score with np.nan in place.
    A low score is determined by the difference between the median score for a given fidcuial marker position
    accross all images and a given score exceeding the threshold.
    """
    df = df.copy()
    for i in np.arange(1, 5):
        fiducials = df.iloc[:, i].values
        corresponding_scores = df.iloc[:, i + 4].values

        median_score = np.median(corresponding_scores)

        for index, value in enumerate(corresponding_scores):
            if median_score - value > threshold:
                fiducials[index] = np.nan
    return df


def nan_offset_fiducial_proxies(
    iter_detect_fiducial_proxies_df, threshold_px=50, missing_proxy=None
):

    df = pd.DataFrame(
        list(iter_detect_fiducial_proxies_df["match_locations"].values),
        columns=["left", "top", "right", "bottom"],
    )
    df.insert(0, "file_names", iter_detect_fiducial_proxies_df["file_names"])
    df = hipp.core.split_position_tuples(df, skip=1)

    for key in df.keys()[1:]:
        offsets = df[key] - np.median(df[key])
        for index, value in enumerate(offsets):
            if abs(value) > threshold_px:  # nan if offset from median position
                df.loc[df.index == index, key] = np.nan

    if missing_proxy == "left":
        df[["left_y", "left_x"]] = (np.nan, np.nan)
    elif missing_proxy == "top":
        df[["top_y", "top_x"]] = (np.nan, np.nan)
    elif missing_proxy == "right":
        df[["right_y", "right_x"]] = (np.nan, np.nan)
    elif missing_proxy == "bottom":
        df[["bottom_y", "bottom_x"]] = (np.nan, np.nan)

    return df


def pad_image(image_array, buffer_distance=250):
    """
    Pad 2D np.array with zeros on all sides.
    """
    a = image_array.shape[0] + 2 * buffer_distance
    b = image_array.shape[1] + 2 * buffer_distance
    padded_img = np.zeros([a, b], dtype=np.uint8)
    padded_img[
        buffer_distance : buffer_distance + image_array.shape[0],
        buffer_distance : buffer_distance + image_array.shape[1],
    ] = image_array
    return padded_img


def slice_image_frame(image_array, windows):

    slices = []
    for window in windows:
        slice_array = image_array[window[0] : window[1], window[2] : window[3]]
        slices.append(slice_array)

    return slices


def split_position_tuples(df, skip=1):
    df = df.copy()
    keys = df.keys().values[skip:]

    for key in keys:
        df_clean = pd.DataFrame(
            df[key].tolist(), index=df.index, columns=[key + "_y", key + "_x"]
        )

        df = pd.concat([df, df_clean], axis=1)

    df = df.drop(keys, axis=1)
    return df


def restitute_single_image(
    image_file: str,
    fiducial_coordinates: np.ndarray,
    principal_point: np.ndarray,
    fiducial_coordinates_true_mm: "np.ndarray | list | None",
    transform_coords: bool,
    transform_image: bool,
    crop_image: bool,
    keep_color: bool,
    qc: bool,
    interpolation_order: int,
    image_square_dim: int,
    output_directory: str,
    scanning_resolution_mm: float,
    geometric_outlier_threshold_px: float = 200.0,
) -> dict:
    """Process a single image: affine-warp to fiducial targets, crop, and compute QC.

    Derives the affine transform that maps detected fiducial positions to their
    calibrated true positions in the image, applies it to the image and/or
    coordinates, crops the result around the transformed principal point, and
    returns per-image QC metrics and log data.

    The calibration coordinates (mm, y-up, origin at PP) are converted to pixel
    targets by dividing by ``scanning_resolution_mm``, flipping Y, and adding the
    detected ``principal_point`` offset — so the affine anchors to the physical
    image frame rather than an arbitrary pixel origin.

    Args:
        image_file: Path to the source image file.
        fiducial_coordinates: Detected fiducial positions in image pixel
            coordinates, shape ``(N, 2)`` ordered as ``[[x0, y0], ...]``.
        principal_point: Detected principal point in image pixel coordinates,
            shape ``(2,)`` as ``[x, y]``.
        fiducial_coordinates_true_mm: True fiducial positions from the camera
            calibration in mm, shape ``(N, 2)`` as ``[[x0, y0], ...]``, where x
            is positive right and y is positive upward with origin at the
            principal point.  Pass ``None`` to skip affine estimation.
        transform_coords: If ``True``, apply the affine transform to the
            fiducial coordinates and principal point.
        transform_image: If ``True``, apply the affine warp to the image array.
        crop_image: If ``True``, crop the warped image to a square centred on
            the transformed principal point.
        keep_color: If ``True``, read and preserve RGB channels; otherwise
            convert to grayscale.
        qc: If ``True``, compute and store QC metrics (RMSE, angle diffs,
            pixel pitch).
        interpolation_order: Spline interpolation order for ``tf.warp``
            (0 = nearest, 1 = bilinear, 3 = bicubic).
        image_square_dim: Side length of the square output crop in pixels.
        output_directory: Directory where the output image is written.
        scanning_resolution_mm: Scanning resolution in mm/px, used to convert
            calibration mm coordinates to pixel targets.

    Returns:
        A flat ``dict`` with keys:

        * ``source_image``, ``output_image``
        * ``principal_point_detected_px``, ``fiducials_detected_px``
        * ``affine_transform`` (``None`` if not estimated)
        * ``principal_point_restituted_px``, ``fiducials_restituted_px``
        * ``principal_point_crop_center_px``, ``crop_bounds_px``
        * ``fiducials_in_crop_px`` — list of ``[x, y]`` per fiducial in the
          cropped image (only populated when ``crop_image=True``)
        * QC metric keys (``NaN`` when not computed)
    """
    # Convert calibration mm coordinates to pixel targets anchored at the
    # detected principal point.  Calibration convention: x right, y up.
    # Image pixel convention: y down → flip Y before adding the PP offset.
    fiducial_coordinates_true = None
    if fiducial_coordinates_true_mm is not None:
        fid_true_mm = np.array(fiducial_coordinates_true_mm, dtype=float)
        fid_true_px = fid_true_mm / scanning_resolution_mm
        fid_true_px[:, 1] *= -1  # y-down image convention
        fiducial_coordinates_true = fid_true_px + principal_point

    result = {
        "source_image": image_file,
        "output_image": None,
        "principal_point_detected_px": principal_point.tolist(),
        "fiducials_detected_px": fiducial_coordinates.tolist(),
        "affine_transform": None,
        "principal_point_restituted_px": None,
        "fiducials_restituted_px": None,
        "principal_point_crop_center_px": None,
        "crop_bounds_px": None,
        # QC metrics — NaN means not computed
        "coordinates_rmse_before_tform_mm": np.nan,
        "coordinates_pp_dist_rmse_before_tform_mm": np.nan,
        "midside_angle_diff_before_tform_mm": np.nan,
        "corner_angle_diff_before_tform": np.nan,
        "coordinates_rmse_after_tform": np.nan,
        "coordinates_pp_dist_rmse_after_tform": np.nan,
        "midside_angle_diff_after_tform": np.nan,
        "corner_angle_diff_after_tform": np.nan,
        "pixel_pitch_after_tform_x": np.nan,
        "pixel_pitch_after_tform_y": np.nan,
    }

    # track true_mm slices so after-transform block can reuse them
    midside_coordinates_true_mm = None
    corner_coordinates_true_mm = None

    # ── QC before transform ───────────────────────────────────────────────────
    if qc and fiducial_coordinates_true_mm is not None:
        # detected fiducials in camera mm frame
        fiducial_coordinates_mm, _ = hipp.qc.convert_coordinates(
            fiducial_coordinates,
            principal_point,
            scanning_resolution_mm=scanning_resolution_mm,
        )
        # true fiducials in camera mm frame (equals calibration values by construction)
        fiducial_coordinates_true_mm_qc, _ = hipp.qc.convert_coordinates(
            fiducial_coordinates_true,
            principal_point,
            scanning_resolution_mm=scanning_resolution_mm,
        )
        result["coordinates_rmse_before_tform_mm"] = hipp.qc.compute_coordinate_rmse(
            fiducial_coordinates_mm, fiducial_coordinates_true_mm_qc
        )
        n_fid = len(fiducial_coordinates_mm)
        if n_fid == 8:
            midside_coordinates_mm = fiducial_coordinates_mm[:4]
            midside_coordinates_true_mm = fiducial_coordinates_true_mm_qc[:4]
            corner_coordinates_mm = fiducial_coordinates_mm[4:]
            corner_coordinates_true_mm = fiducial_coordinates_true_mm_qc[4:]
            result["midside_angle_diff_before_tform_mm"] = hipp.qc.compute_angle_diff(
                midside_coordinates_mm, midside_coordinates_true_mm
            )
            result["corner_angle_diff_before_tform"] = hipp.qc.compute_angle_diff(
                corner_coordinates_mm, corner_coordinates_true_mm
            )
            result["coordinates_pp_dist_rmse_before_tform_mm"] = (
                hipp.qc.compute_coordinate_distance_diff_rmse(
                    midside_coordinates_mm,
                    midside_coordinates_true_mm,
                    corner_coordinates_mm,
                    corner_coordinates_true_mm,
                )
            )
        elif n_fid == 4:
            midside_coordinates_mm = fiducial_coordinates_mm[:4]
            midside_coordinates_true_mm = fiducial_coordinates_true_mm_qc[:4]
            result["midside_angle_diff_before_tform_mm"] = hipp.qc.compute_angle_diff(
                midside_coordinates_mm, midside_coordinates_true_mm
            )
            result["coordinates_pp_dist_rmse_before_tform_mm"] = (
                hipp.qc.compute_coordinate_distance_diff_rmse(
                    midside_coordinates_mm, midside_coordinates_true_mm, None, None
                )
            )

    # ── Load image ────────────────────────────────────────────────────────────
    image_array = None
    if transform_image or crop_image:
        flags = cv2.IMREAD_COLOR if keep_color else cv2.IMREAD_GRAYSCALE
        image_array = cv2.imread(image_file, flags)

    # ── Affine transform ──────────────────────────────────────────────────────
    fiducial_coordinates_tform = fiducial_coordinates.copy()
    if (transform_image or transform_coords) and fiducial_coordinates_true is not None:
        # Keep only rows where both detected and true coords are valid
        valid = ~(
            np.isnan(fiducial_coordinates).any(axis=1)
            | np.isnan(fiducial_coordinates_true).any(axis=1)
        )
        # Geometric consistency filter: reject detections whose distance from
        # the principal point deviates too much from the calibrated distance.
        if geometric_outlier_threshold_px > 0:
            expected_dist = np.linalg.norm(
                fiducial_coordinates_true - principal_point, axis=1
            )
            detected_dist = np.linalg.norm(
                fiducial_coordinates - principal_point, axis=1
            )
            geometric_outliers = (
                np.abs(detected_dist - expected_dist) > geometric_outlier_threshold_px
            )
            if geometric_outliers.any():
                print(
                    f"[geometric filter] {image_file}: rejecting fiducial(s) at "
                    f"index {np.where(geometric_outliers)[0].tolist()} "
                    f"(deviation > {geometric_outlier_threshold_px:.0f} px)"
                )
            valid &= ~geometric_outliers

        fid_valid = fiducial_coordinates[valid]
        fid_true_valid = fiducial_coordinates_true[valid]

        if len(fid_valid) >= 3:  # Need at least 3 points for affine transform
            tform = tf.estimate_transform("affine", fid_valid, fid_true_valid)

            fiducial_coordinates_tform = tform(fiducial_coordinates)
            principal_point = tform(principal_point)[0]

            result["pixel_pitch_after_tform_x"] = float(
                np.round(tform.scale[1], 4) * scanning_resolution_mm
            )
            result["pixel_pitch_after_tform_y"] = float(
                np.round(tform.scale[0], 4) * scanning_resolution_mm
            )
            result["affine_transform"] = {
                "matrix_3x3": tform.params.tolist(),
                "translation_px": list(tform.translation),
                "rotation_rad": float(tform.rotation),
                "scale": list(tform.scale),
                "shear_rad": float(tform.shear),
            }
            result["principal_point_restituted_px"] = principal_point.tolist()
            result["fiducials_restituted_px"] = fiducial_coordinates_tform.tolist()

            if transform_image and image_array is not None:
                A = np.linalg.inv(tform.params)
                image_array = (
                    tf.warp(
                        image_array,
                        A,
                        output_shape=image_array.shape,
                        order=interpolation_order,
                    )
                    * 255
                ).astype(np.uint8)

            # ── QC after transform ────────────────────────────────────────────
            if qc and fiducial_coordinates_true_mm is not None:
                fiducial_coordinates_tform_mm, _ = hipp.qc.convert_coordinates(
                    fiducial_coordinates_tform,
                    principal_point,
                    scanning_resolution_mm=scanning_resolution_mm,
                )
                result["coordinates_rmse_after_tform"] = (
                    hipp.qc.compute_coordinate_rmse(
                        fiducial_coordinates_tform_mm, fiducial_coordinates_true_mm
                    )
                )
                n_fid = len(fiducial_coordinates_tform_mm)
                if n_fid == 8:
                    midside_tform_mm = fiducial_coordinates_tform_mm[:4]
                    corner_tform_mm = fiducial_coordinates_tform_mm[4:]
                    result["midside_angle_diff_after_tform"] = (
                        hipp.qc.compute_angle_diff(
                            midside_tform_mm, midside_coordinates_true_mm
                        )
                    )
                    result["corner_angle_diff_after_tform"] = (
                        hipp.qc.compute_angle_diff(
                            corner_tform_mm, corner_coordinates_true_mm
                        )
                    )
                    result["coordinates_pp_dist_rmse_after_tform"] = (
                        hipp.qc.compute_coordinate_distance_diff_rmse(
                            midside_tform_mm,
                            midside_coordinates_true_mm,
                            corner_tform_mm,
                            corner_coordinates_true_mm,
                        )
                    )
                elif n_fid == 4:
                    midside_tform_mm = fiducial_coordinates_tform_mm[:4]
                    result["midside_angle_diff_after_tform"] = (
                        hipp.qc.compute_angle_diff(
                            midside_tform_mm, midside_coordinates_true_mm
                        )
                    )
                    result["coordinates_pp_dist_rmse_after_tform"] = (
                        hipp.qc.compute_coordinate_distance_diff_rmse(
                            midside_tform_mm, midside_coordinates_true_mm, None, None
                        )
                    )

    # ── Crop and write ────────────────────────────────────────────────────────
    if crop_image and image_array is not None:
        principal_point_int = np.array([int(round(v)) for v in principal_point])
        half = int(round(image_square_dim / 2))
        pp_x, pp_y = int(principal_point_int[0]), int(principal_point_int[1])
        result["principal_point_crop_center_px"] = [pp_x, pp_y]
        result["crop_bounds_px"] = {
            "x_left": pp_x - half,
            "x_right": pp_x + half,
            "y_top": pp_y - half,
            "y_bottom": pp_y + half,
        }
        # Fiducial positions in the cropped image: subtract the crop offset
        # (crop_center - half) from the warped coordinates.
        result["fiducials_in_crop_px"] = [
            [fid[0] - (pp_x - half), fid[1] - (pp_y - half)]
            for fid in fiducial_coordinates_tform
        ]
        image_array = hipp.image.crop_about_point(
            image_array,
            principal_point_int[::-1],  # crop_about_point expects y, x
            image_square_dim=image_square_dim,
        )
        _, basename, extension = hipp.io.split_file(image_file)
        out = os.path.join(output_directory, basename + extension)
        cv2.imwrite(out, image_array)
        result["output_image"] = out
        print(out)

    elif transform_image and image_array is not None:
        _, basename, extension = hipp.io.split_file(image_file)
        out = os.path.join(output_directory, basename + extension)
        cv2.imwrite(out, image_array)
        result["output_image"] = out

    return result
