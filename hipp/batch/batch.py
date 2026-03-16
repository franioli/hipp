import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage import transform as tf

import hipp.core
import hipp.image
import hipp.io
import hipp.plot
import hipp.qc


def compute_safe_image_square_dim(
    df: pd.DataFrame,
    image_file_column: str = "fileName",
    principal_point_x_column: str = "principal_point_x",
    principal_point_y_column: str = "principal_point_y",
) -> tuple[int, list[int]]:
    """Compute safe square crop dimensions for a set of images.

    For each image this computes the largest square (same half-distance in x and y)
    that can be centered on the detected principal point without going out of image bounds.
    Returns the minimum safe square dimension across all images and the list of per-image sizes.

    Images with a missing (NaN) principal point are skipped and get a safe_dim of 0.
    A warning is printed for each such image; if ALL images are skipped, a ValueError
    is raised because no usable crop dimension can be determined.

    Args:
        df: DataFrame containing at least the image file path and principal point columns.
        image_file_column: Name of the column with the image file path.
        principal_point_x_column: Name of the principal point X column (pixel coordinate).
        principal_point_y_column: Name of the principal point Y column (pixel coordinate).

    Returns:
        A tuple (min_safe_dim, per_image_safe_dims) where:
        - min_safe_dim is the smallest safe square dimension (int) that fits every image
          that has a valid principal point.
        - per_image_safe_dims is a list of ints with the safe square dimension per image
          (0 for images with a missing or out-of-bounds principal point).

    Raises:
        FileNotFoundError: if an image path does not exist.
        IOError: if an image cannot be read.
        ValueError: if no valid safe dimensions could be computed.
    """
    safe_dims: list[int] = []

    for _, row in df.iterrows():
        img_path = Path(row[image_file_column])
        img_name = img_path.name

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise OSError(f"Failed to read image: {img_path}")

        h, w = img.shape
        pp_y = row[principal_point_y_column]
        pp_x = row[principal_point_x_column]

        # Missing principal point — skip gracefully
        if pd.isna(pp_y) or pd.isna(pp_x):
            print(
                f"  Warning: {img_name} — principal point is NaN, skipping (safe_dim=0)"
            )
            safe_dims.append(0)
            continue

        pp_y, pp_x = float(pp_y), float(pp_x)

        # Principal point outside image bounds
        if pp_x <= 0 or pp_x >= w or pp_y <= 0 or pp_y >= h:
            print(
                f"  Warning: {img_name} — principal point ({pp_x:.1f}, {pp_y:.1f}) "
                f"is outside image bounds ({w}×{h}), skipping (safe_dim=0)"
            )
            safe_dims.append(0)
            continue

        max_half = min(pp_y, h - pp_y, pp_x, w - pp_x)
        safe_dims.append(int(max_half) * 2 if max_half > 0 else 0)

    valid_dims = [d for d in safe_dims if d > 0]
    if not valid_dims:
        raise ValueError(
            "No valid safe image square dimensions could be computed. "
            "Check that principal points were detected correctly."
        )

    n_skipped = len(safe_dims) - len(valid_dims)
    if n_skipped:
        print(
            f"\n  {n_skipped}/{len(safe_dims)} image(s) had no valid principal point "
            f"and are excluded from the min_safe_dim calculation."
        )

    min_safe_dim = min(valid_dims)
    return min_safe_dim, safe_dims


def image_restitution(
    df_detected,
    fiducial_coordinates_true_mm=None,
    image_file_name_column_name="fileName",
    scanning_resolution_mm=0.02,
    transform_coords=True,
    transform_image=True,
    crop_image=True,
    image_square_dim=10800,
    interpolation_order=3,
    output_directory="input_data/preprocessed_images/",
    qc=True,
    keep_color=False,
):
    """
    Computes affine transformation between detected coordinates and true coordinates,
    then transforms image array.

    Parameters
    ----------
    df_detected : pd.DataFrame
        DataFrame with detected fiducial coordinates and image file names.
    fiducial_coordinates_true_mm : list or array-like, optional
        True fiducial coordinates in mm (x, y pairs).
    image_file_name_column_name : str, default="fileName"
        Column name containing image file paths.
    scanning_resolution_mm : float, default=0.02
        Scanning resolution in mm/px.
    transform_coords : bool, default=True
        Whether to transform coordinates using affine transformation.
    transform_image : bool, default=True
        Whether to transform the image array.
    crop_image : bool, default=True
        Whether to crop the image around the principal point.
    image_square_dim : int, default=10800
        Size of the square crop in pixels.
    interpolation_order : int, default=3
        Interpolation order for warp (0=nearest, 1=bilinear, 3=cubic).
    output_directory : str, default="input_data/preprocessed_images/"
        Output directory for transformed images.
    qc : bool, default=True
        Whether to generate quality control metrics and logs.
    keep_color : bool, default=False
        If False, convert images to grayscale. If True, keep all RGB channels.

    Notes
    -----
    Interpolation orders:
    0: Nearest-neighbor
    1: Bi-linear
    2: Bi-quadratic
    3: Bi-cubic
    4: Bi-quartic
    5: Bi-quintic
    """

    # TODO add logging

    if transform_image or crop_image:
        p = Path(output_directory)
        p.mkdir(parents=True, exist_ok=True)

    # QC lists
    coordinates_rmse_before_tform = []
    coordinates_rmse_after_tform = []
    coordinates_pp_dist_rmse_before_tform = []
    coordinates_pp_dist_rmse_after_tform = []
    midside_angle_diff_before_tform = []
    midside_angle_diff_after_tform = []
    corner_angle_diff_before_tform = []
    corner_angle_diff_after_tform = []
    pixel_pitches = []
    qc_dataframes = []

    # ← LOG: global processing record
    if qc:
        processing_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parameters": {
                "scanning_resolution_mm": scanning_resolution_mm,
                "image_square_dim": image_square_dim,
                "interpolation_order": interpolation_order,
                "transform_coords": transform_coords,
                "transform_image": transform_image,
                "crop_image": crop_image,
                "output_directory": str(output_directory),
                "fiducial_coordinates_true_mm": (
                    fiducial_coordinates_true_mm.tolist()
                    if hasattr(fiducial_coordinates_true_mm, "tolist")
                    else fiducial_coordinates_true_mm
                ),
            },
            "images": [],
        }

    # convert true coordinates to image reference system
    if fiducial_coordinates_true_mm is not None:
        fiducial_coordinates_true_mm = np.array(
            fiducial_coordinates_true_mm, dtype=float
        )
        fiducial_coordinates_true_px = (
            fiducial_coordinates_true_mm / scanning_resolution_mm
        )
        fiducial_coordinates_true_px[:, 1] = fiducial_coordinates_true_px[:, 1] * -1

    # prepare dataframe with detected coordinates
    df_coords = df_detected.drop(
        [image_file_name_column_name, "principal_point_x", "principal_point_y"], axis=1
    )

    for index, row in df_coords.iterrows():
        # convert coordinates to x,y order
        fiducial_coordinates = np.array(list(zip(row.values[0::2], row.values[1::2])))
        fiducial_coordinates = fiducial_coordinates[:, ::-1]

        # extract principal point
        principal_point = np.array(
            (
                df_detected["principal_point_x"].iloc[index],
                df_detected["principal_point_y"].iloc[index],
            )
        )

        # ← LOG: capture detected state before any transform
        pp_detected = principal_point.copy()
        fid_detected = fiducial_coordinates.copy()

        # add principal point to get true fiducial coordinates into image reference system
        if fiducial_coordinates_true_mm is not None:
            fiducial_coordinates_true = fiducial_coordinates_true_px + principal_point

        if qc and fiducial_coordinates_true_mm is not None:
            fiducial_coordinates_mm, principal_point_mm = hipp.qc.convert_coordinates(
                fiducial_coordinates,
                principal_point,
                scanning_resolution_mm=scanning_resolution_mm,
            )
            fiducial_coordinates_true_mm, _ = hipp.qc.convert_coordinates(
                fiducial_coordinates_true,
                principal_point,
                scanning_resolution_mm=scanning_resolution_mm,
            )
            rmse = hipp.qc.compute_coordinate_rmse(
                fiducial_coordinates_mm, fiducial_coordinates_true_mm
            )
            coordinates_rmse_before_tform.append(rmse)

            if len(fiducial_coordinates_mm) == 8:
                midside_coordinates_mm = fiducial_coordinates_mm[:4]
                midside_coordinates_true_mm = fiducial_coordinates_true_mm[:4]
                corner_coordinates_mm = fiducial_coordinates_mm[4:]
                corner_coordinates_true_mm = fiducial_coordinates_true_mm[4:]
                diff = hipp.qc.compute_angle_diff(
                    midside_coordinates_mm, midside_coordinates_true_mm
                )
                midside_angle_diff_before_tform.append(diff)
                diff = hipp.qc.compute_angle_diff(
                    corner_coordinates_mm, corner_coordinates_true_mm
                )
                corner_angle_diff_before_tform.append(diff)
                rmse = hipp.qc.compute_coordinate_distance_diff_rmse(
                    midside_coordinates_mm,
                    midside_coordinates_true_mm,
                    corner_coordinates_mm,
                    corner_coordinates_true_mm,
                )
                coordinates_pp_dist_rmse_before_tform.append(rmse)

            elif len(fiducial_coordinates_mm) == 4:
                midside_coordinates_mm = fiducial_coordinates_mm[:4]
                midside_coordinates_true_mm = fiducial_coordinates_true_mm[:4]
                diff = hipp.qc.compute_angle_diff(
                    midside_coordinates_mm, midside_coordinates_true_mm
                )
                midside_angle_diff_before_tform.append(diff)
                rmse = hipp.qc.compute_coordinate_distance_diff_rmse(
                    midside_coordinates_mm, midside_coordinates_true_mm, None, None
                )

        if transform_image or crop_image:
            image_file = df_detected[image_file_name_column_name].iloc[index]
            if keep_color:
                image_array = cv2.imread(image_file, cv2.IMREAD_COLOR)
            else:
                image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

        # ← LOG: initialise per-image entry
        if qc:
            image_log = {
                "source_image": str(image_file)
                if (transform_image or crop_image)
                else None,
                "output_image": None,
                "principal_point_detected_px": pp_detected.tolist(),
                "fiducials_detected_px": fid_detected.tolist(),
                "affine_transform": None,
                "principal_point_restituted_px": None,
                "principal_point_crop_center_px": None,
                "crop_bounds_px": None,
            }

        if transform_image or transform_coords:
            fid_coord_tmp = np.where(
                ~np.isnan(fiducial_coordinates_true), fiducial_coordinates, np.nan
            )
            fid_coord_true_tmp = np.where(
                ~np.isnan(fiducial_coordinates), fiducial_coordinates_true, np.nan
            )
            fid_coord_tmp = np.array(
                [x for x in fid_coord_tmp if ~np.isnan(x).any()], dtype=float
            )
            fid_coord_true_tmp = np.array(
                [x for x in fid_coord_true_tmp if ~np.isnan(x).any()], dtype=float
            )

            if len(fid_coord_tmp) >= 3 and ~np.isnan(fid_coord_true_tmp).all():
                tform = tf.AffineTransform()
                tform.estimate(fid_coord_tmp, fid_coord_true_tmp)

                fiducial_coordinates_tform = tform(fiducial_coordinates)
                principal_point = tform(principal_point)[0]

                pixel_pitch_x_tmp = np.round(tform.scale[1], 4)
                pixel_pitch_y_tmp = np.round(tform.scale[0], 4)
                pixel_pitch_tmp = (
                    pixel_pitch_x_tmp * scanning_resolution_mm,
                    pixel_pitch_y_tmp * scanning_resolution_mm,
                )
                pixel_pitches.append(pixel_pitch_tmp)

                if qc:
                    # ← LOG: affine parameters and restituted principal point
                    image_log["affine_transform"] = {
                        "matrix_3x3": tform.params.tolist(),
                        "translation_px": list(tform.translation),
                        "rotation_rad": float(tform.rotation),
                        "scale": list(tform.scale),
                        "shear_rad": float(tform.shear),
                    }
                    image_log["principal_point_restituted_px"] = (
                        principal_point.tolist()
                    )
                    image_log["fiducials_restituted_px"] = (
                        fiducial_coordinates_tform.tolist()
                    )

                if transform_image:
                    A = np.linalg.inv(tform.params)
                    image_array_transformed = tf.warp(
                        image_array,
                        A,
                        output_shape=image_array.shape,
                        order=interpolation_order,
                    )
                    image_array = (image_array_transformed * 255).astype(np.uint8)

                if qc:
                    fiducial_coordinates_tform_mm, principal_point_tform_mm = (
                        hipp.qc.convert_coordinates(
                            fiducial_coordinates_tform,
                            principal_point,
                            scanning_resolution_mm=scanning_resolution_mm,
                        )
                    )
                    rmse = hipp.qc.compute_coordinate_rmse(
                        fiducial_coordinates_tform_mm, fiducial_coordinates_true_mm
                    )
                    coordinates_rmse_after_tform.append(rmse)

                    if len(fiducial_coordinates_tform_mm) == 8:
                        midside_coordinates_tform_mm = fiducial_coordinates_tform_mm[:4]
                        corner_coordinates_tform_mm = fiducial_coordinates_tform_mm[4:]
                        diff = hipp.qc.compute_angle_diff(
                            midside_coordinates_tform_mm, midside_coordinates_true_mm
                        )
                        midside_angle_diff_after_tform.append(diff)
                        diff = hipp.qc.compute_angle_diff(
                            corner_coordinates_tform_mm, corner_coordinates_true_mm
                        )
                        corner_angle_diff_after_tform.append(diff)
                        rmse = hipp.qc.compute_coordinate_distance_diff_rmse(
                            midside_coordinates_tform_mm,
                            midside_coordinates_true_mm,
                            corner_coordinates_tform_mm,
                            corner_coordinates_true_mm,
                        )
                        coordinates_pp_dist_rmse_after_tform.append(rmse)
                    elif len(fiducial_coordinates_tform_mm) == 4:
                        midside_coordinates_tform_mm = fiducial_coordinates_tform_mm[:4]
                        diff = hipp.qc.compute_angle_diff(
                            midside_coordinates_tform_mm, midside_coordinates_true_mm
                        )
                        midside_angle_diff_after_tform.append(diff)
                        rmse = hipp.qc.compute_coordinate_distance_diff_rmse(
                            midside_coordinates_tform_mm,
                            midside_coordinates_true_mm,
                            None,
                            None,
                        )
                        coordinates_pp_dist_rmse_after_tform.append(rmse)

        if crop_image:
            principal_point = np.array([int(round(x)) for x in principal_point])

            # ← LOG: crop geometry (mirrors crop_about_point logic)
            if qc:
                half = int(round(image_square_dim / 2))
                pp_x, pp_y = int(principal_point[0]), int(principal_point[1])
                image_log["principal_point_crop_center_px"] = [pp_x, pp_y]
                image_log["crop_bounds_px"] = {
                    "x_left": pp_x - half,
                    "x_right": pp_x + half,
                    "y_top": pp_y - half,
                    "y_bottom": pp_y + half,
                }

            image_array = hipp.image.crop_about_point(
                image_array,
                principal_point[::-1],  # requires y,x order
                image_square_dim=image_square_dim,
            )
            path, basename, extension = hipp.io.split_file(image_file)
            out = os.path.join(output_directory, basename + extension)
            cv2.imwrite(out, image_array)
            print(out)

        elif transform_image:
            path, basename, extension = hipp.io.split_file(image_file)
            out = os.path.join(output_directory, basename + extension)
            cv2.imwrite(out, image_array)

    if qc:
        # ← LOG: record output path and append
        if transform_image or crop_image:
            image_log["output_image"] = out
        processing_log["images"].append(image_log)

        # ← LOG: save JSON alongside outputs
        log_path = os.path.join(output_directory, "restitution_log.json")
        with open(log_path, "w") as f:
            json.dump(processing_log, f, indent=2)
        print(f"Processing log saved to {log_path}")

        qc_dataframes = []

        qc_dataframes.append(
            pd.DataFrame(
                list(df_detected[image_file_name_column_name].values),
                columns=[image_file_name_column_name],
            )
        )
        qc_dataframes.append(
            pd.DataFrame(
                coordinates_rmse_before_tform, columns=["coordinates_rmse_before_tform"]
            )
        )
        qc_dataframes.append(
            pd.DataFrame(
                coordinates_pp_dist_rmse_before_tform,
                columns=["coordinates_pp_dist_rmse_before_tform"],
            )
        )
        qc_dataframes.append(
            pd.DataFrame(
                midside_angle_diff_before_tform,
                columns=["midside_angle_diff_before_tform"],
            )
        )
        qc_dataframes.append(
            pd.DataFrame(
                corner_angle_diff_before_tform,
                columns=["corner_angle_diff_before_tform"],
            )
        )
        if transform_coords:
            qc_dataframes.append(
                pd.DataFrame(
                    pixel_pitches,
                    columns=["pixel_pitch_after_tform_x", "pixel_pitch_after_tform_y"],
                )
            )
            qc_dataframes.append(
                pd.DataFrame(
                    coordinates_rmse_after_tform,
                    columns=["coordinates_rmse_after_tform"],
                )
            )
            qc_dataframes.append(
                pd.DataFrame(
                    coordinates_pp_dist_rmse_after_tform,
                    columns=["coordinates_pp_dist_rmse_after_tform"],
                )
            )
            qc_dataframes.append(
                pd.DataFrame(
                    midside_angle_diff_after_tform,
                    columns=["midside_angle_diff_after_tform"],
                )
            )
            qc_dataframes.append(
                pd.DataFrame(
                    corner_angle_diff_after_tform,
                    columns=["corner_angle_diff_after_tform"],
                )
            )

            qc_df = pd.concat(qc_dataframes, axis=1)
            qc_df.index = qc_df[image_file_name_column_name].str[-12:-4]

            hipp.plot.plot_restitution_qc(qc_df)


def iter_detect_fiducials(
    image_files_directory="input_data/raw_images/",
    image_file_name_column_name="fileName",
    image_files_extension=".tif",
    template_file=None,
    template_high_res_zoomed_file=None,
    midside_fiducials=False,
    corner_fiducials=False,
    center_fiducial=False,
    qc=True,
):
    """
    Function to iteratively detect fiducial markers in a set of images and return as pandas.DataFrame.

    Ensure that the templates correspond to either the fiducial markers at the midside or corners.
    Specify flag accordingly.
    """

    images = sorted(
        glob.glob(os.path.join(image_files_directory, "*" + image_files_extension))
    )

    print(f"\n{'=' * 70}")
    print("[iter_detect_fiducials] Starting fiducial detection")
    print(f"  Images directory: {image_files_directory}")
    print(f"  Found {len(images)} images with extension '{image_files_extension}'")
    print(f"  Template file: {template_file}")
    if template_high_res_zoomed_file:
        print(f"  High-res template: {template_high_res_zoomed_file}")

    template_array = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
    fiducial_locations = []
    intersection_angles = []
    principal_points = []
    quality_scores = []

    for idx, image_file in enumerate(images, 1):
        print(f"  [{idx}/{len(images)}] Processing: {os.path.basename(image_file)}")
        image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

        # Subset image array into window slices to speed up template matching
        if midside_fiducials:
            windows = hipp.core.define_midside_windows(image_array)

        elif corner_fiducials:
            windows = hipp.core.define_corner_windows(image_array)

        elif center_fiducial:
            windows = hipp.core.define_center_window(image_array)

        else:
            print(
                "Please specify midside or corner fiducials and provide corresponding templates."
            )
            break

        slices = hipp.core.slice_image_frame(image_array, windows)

        # Detect fiducial in each window
        matches, qs = hipp.core.detect_fiducials(slices, template_array, windows)
        print(f"        → Coarse detection: {len(matches)} fiducials detected")

        if midside_fiducials:
            labels = ["midside_left", "midside_top", "midside_right", "midside_bottom"]

        elif corner_fiducials:
            labels = [
                "corner_top_left",
                "corner_top_right",
                "corner_bottom_right",
                "corner_bottom_left",
            ]

        elif center_fiducial:
            labels = ["principal_point"]

        quality_score_labels = [sub + "_score" for sub in labels]

        template_array = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
        distance_from_loc = template_array.shape[0]  # = 2 × distance_around_fiducial

        subpixel_fiducial_locations, subpixel_quality_scores = (
            hipp.core.detect_subpixel_fiducial_coordinates(
                image_file,
                image_array,
                matches,
                template_high_res_zoomed_file,
                labels=labels,
                distance_from_loc=distance_from_loc,
                qc=qc,
            )
        )
        print(
            f"        → Subpixel refinement: {len(subpixel_fiducial_locations)} fiducials refined"
        )

        fiducial_locations.append(subpixel_fiducial_locations)
        quality_scores.append(subpixel_quality_scores)

    images_df = pd.DataFrame(images, columns=[image_file_name_column_name])
    fiducial_locations_df = pd.DataFrame(fiducial_locations, columns=labels)
    quality_scores_df = pd.DataFrame(quality_scores, columns=quality_score_labels)

    if not center_fiducial:
        principal_points_df = hipp.core.compute_principal_points(
            fiducial_locations_df, quality_scores_df
        )
        df = pd.concat(
            [images_df, fiducial_locations_df, quality_scores_df, principal_points_df],
            axis=1,
        )

    if center_fiducial:
        df = pd.concat([images_df, fiducial_locations_df, quality_scores_df], axis=1)

    print(f"\n{'=' * 70}")
    print("[iter_detect_fiducials] Detection complete")
    print(f"  Total images processed: {len(images)}")
    print(f"  Output DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"{'=' * 70}\n")

    return df


def preprocess_with_fiducial_proxies(
    image_directory,
    template_directory,
    buffer_distance=250,
    threshold_px=50,
    stretch_histogram=True,
    clahe_enhancement=True,
    image_square_dim=None,
    output_directory="input_data/cropped_images",
    verbose=True,
    missing_proxy=None,
    qc_df=True,
    qc_df_output_directory="qc/proxy_detection_data_frames",
    qc_plots=True,
    qc_plots_output_directory="qc/proxy_detection",
    EE_find_matching_template=False,
    max_workers=None,
):
    """
    Detects fiducial marker proxies at midside left, top, right, and bottom positions.

    Buffers image with zero values in order to enable moving template over proxy at
    image edge for matching.

    Requires at least two fiducial marker proxies to approximate principal point.

    To read in and examine QC dataframe use pandas.read_pickle('proxy_locations_df.pd'),
    for example.
    """
    images = sorted(Path(image_directory).glob("*tif"))

    if EE_find_matching_template:
        detected_df_list = []
        proxy_locations_df_list = []
        principal_points_list = []
        distances_list = []
        intersection_angles_list = []

        # find matching EE template based on roll name
        rolls = sorted(set([Path(i).stem[:-4] for i in images]))
        template_dirs = [t for t in Path(template_directory).iterdir() if t.is_dir()]

        for r in rolls:
            template_dir = None
            images_tmp = [img for img in images if r in img.stem]
            for t in template_dirs:
                if r in t.as_posix():
                    template_dir = t.as_posix()
            if not template_dir:
                print(
                    "No matching templates found for",
                    r,
                    "in provided template directory",
                )
                sys.exit(1)

            print("Templates found for roll", r)
            images_tmp = [img.as_posix() for img in images if r in img.stem]
            templates = hipp.core.load_midside_fiducial_proxy_templates(template_dir)
            detected_df = hipp.core.iter_detect_fiducial_proxies(
                images_tmp, templates, buffer_distance=buffer_distance, verbose=verbose
            )

            proxy_locations_df = hipp.core.nan_offset_fiducial_proxies(
                detected_df, threshold_px=threshold_px, missing_proxy=missing_proxy
            )

            result = hipp.core.compute_principal_point_from_proxies(
                proxy_locations_df, verbose=verbose
            )

            principal_points, distances, intersection_angles = result

            if np.isnan(np.nanmin(distances)):
                print("""Could not compute distance between any fiducial proxies and principal point. 
                Detection likely failed. Check your inputs.""")
                sys.exit(1)

            image_square_dim = (
                int(round((np.nanmin(distances)) / 2)) * 2
            )  # ensure half is non float for array index slicing

            new_image_square_dim = hipp.core.validate_square_dim(
                images_tmp, buffer_distance, principal_points, image_square_dim
            )

            if new_image_square_dim and missing_proxy:
                print("Missing_proxy set to", missing_proxy)
                print(
                    "Adjusting final image dimensions to minimum viable size from",
                    image_square_dim,
                    "to",
                    new_image_square_dim,
                )
                print("Check qc plots for:")
                for i in images_tmp:
                    print(i)
                image_square_dim = new_image_square_dim

            elif new_image_square_dim:
                msg = "\n".join(
                    [
                        "WARNING: Irregular final image dimensions detected",
                        "Likely due to missing fiducial marker on a side of the images.",
                        "Check qc plots for:",
                    ]
                )
                print(msg)
                for i in images_tmp:
                    print(i)
                if not missing_proxy:
                    msg = "\n".join(
                        [
                            "Consider reprocessing by setting missing_proxy option to left, top, right, or bottom",
                            "to improve principal point detection.",
                        ]
                    )
                    print(msg)

            print("Cropping images to square with dimensions", str(image_square_dim))
            hipp.core.iter_crop_image_from_file(
                images_tmp,
                principal_points,
                image_square_dim,
                output_directory=output_directory,
                buffer_distance=buffer_distance,
                stretch_histogram=stretch_histogram,
                clahe_enhancement=clahe_enhancement,
                verbose=verbose,
            )

            if qc_plots:
                print("Plotting proxy detection QC plots at", qc_plots_output_directory)
                hipp.plot.iter_plot_proxies(
                    images_tmp,
                    proxy_locations_df,
                    principal_points,
                    buffer_distance=buffer_distance,
                    output_directory=qc_plots_output_directory,
                    verbose=verbose,
                )

            detected_df_list.append(detected_df)
            proxy_locations_df_list.append(proxy_locations_df)
            principal_points_list.extend(principal_points)
            distances_list.extend(distances)
            intersection_angles_list.extend(intersection_angles)

        detected_df = pd.concat(detected_df_list)
        proxy_locations_df = pd.concat(proxy_locations_df_list)
        principal_points = principal_points_list
        distances = distances_list
        intersection_angles = intersection_angles_list

    else:
        images = [img.as_posix() for img in images]
        templates = hipp.core.load_midside_fiducial_proxy_templates(template_directory)
        detected_df = hipp.core.iter_detect_fiducial_proxies(
            images,
            templates,
            buffer_distance=buffer_distance,
            max_workers=max_workers,
            verbose=verbose,
        )

        proxy_locations_df = hipp.core.nan_offset_fiducial_proxies(
            detected_df, threshold_px=threshold_px, missing_proxy=missing_proxy
        )
        result = hipp.core.compute_principal_point_from_proxies(
            proxy_locations_df, verbose=verbose
        )
        principal_points, distances, intersection_angles = result
        min_dist = np.nanmin(distances)
        if np.isnan(min_dist):
            print(
                f"ERROR: Could not compute distance between any fiducial proxies. "
                f"Detection likely failed for all images.\n"
                f"Possible causes:\n"
                f"  - The templates do not match the fiducial markers in the images.\n"
                f"  - The threshold_px value is too strict (currently {threshold_px} px).\n"
                f"  - The buffer_distance is incorrect.\n"
                f"Check that your template images (L.tif, T.tif, R.tif, B.tif) look correct "
                f"and that the fiducial markers are visible in the input images."
            )
            return None

        image_square_dim = (
            int(round(min_dist / 2)) * 2
        )  # ensure half is non float for array index slicing

        new_image_square_dim = hipp.core.validate_square_dim(
            images, buffer_distance, principal_points, image_square_dim
        )
        if new_image_square_dim and missing_proxy:
            print("Missing_proxy set to", missing_proxy)
            print(
                "Adjusting final image dimensions to minimum viable size from",
                image_square_dim,
                "to",
                new_image_square_dim,
            )
            print("Check qc plots for:")
            for i in images:
                print(i)
            image_square_dim = new_image_square_dim

        elif new_image_square_dim:
            msg = "\n".join(
                [
                    "WARNING: Irregular final image dimensions detected",
                    "Likely due to missing fiducial marker on a side of the images.",
                    "Check qc plots for:",
                ]
            )
            print(msg)
            for i in images:
                print(i)
            if not missing_proxy:
                msg = "\n".join(
                    [
                        "Consider reprocessing by setting missing_proxy option to left, top, right, or bottom",
                        "to improve principal point detection.",
                    ]
                )
                print(msg)

        print("Cropping images to square with dimensions", str(image_square_dim))
        hipp.core.iter_crop_image_from_file(
            images,
            principal_points,
            image_square_dim,
            output_directory=output_directory,
            buffer_distance=buffer_distance,
            stretch_histogram=stretch_histogram,
            clahe_enhancement=clahe_enhancement,
            verbose=verbose,
        )
        if qc_plots:
            print("Plotting proxy detection QC plots at", qc_plots_output_directory)
            hipp.plot.iter_plot_proxies(
                images,
                proxy_locations_df,
                principal_points,
                buffer_distance=buffer_distance,
                output_directory=qc_plots_output_directory,
                verbose=verbose,
            )
    if qc_df:
        print("Saving proxy detection QC dataframes to", qc_df_output_directory)
        p = Path(qc_df_output_directory)
        p.mkdir(parents=True, exist_ok=True)
        detected_df.to_pickle(os.path.join(qc_df_output_directory, "detected_df.pd"))
        proxy_locations_df.to_pickle(
            os.path.join(qc_df_output_directory, "proxy_locations_df.pd")
        )
        pd.DataFrame(principal_points).to_pickle(
            os.path.join(qc_df_output_directory, "principal_points.pd")
        )
        pd.DataFrame(distances).to_pickle(
            os.path.join(qc_df_output_directory, "distances.pd")
        )
        pd.DataFrame(intersection_angles).to_pickle(
            os.path.join(qc_df_output_directory, "intersection_angles.pd")
        )

    return image_square_dim
