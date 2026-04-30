import concurrent.futures
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import transform as tf

import hipp.core
import hipp.image
import hipp.io
import hipp.plot
import hipp.qc

Image.MAX_IMAGE_PIXELS = None  # aerial scans legitimately exceed PIL's default limit


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
        try:
            with Image.open(img_path) as _img:
                w, h = _img.size  # PIL returns (width, height), no pixel data loaded
        except Exception as exc:
            safe_dims.append(0)
            print(f"  Error reading {img_name}: {exc}. Skipping (safe_dim=0)")
            continue

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


def _detect_single_image(args):
    (
        image_file,
        template_file,
        template_high_res_zoomed_file,
        midside_fiducials,
        corner_fiducials,
        center_fiducial,
        labels,
        distance_from_loc,
        subpx_upsample_factor,
        qc,
        qc_directory,
        idx,
        total,
    ) = args

    print(f"  [{idx}/{total}] Processing: {os.path.basename(image_file)}")
    image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    if midside_fiducials:
        windows = hipp.core.define_midside_windows(image_array)
    elif corner_fiducials:
        windows = hipp.core.define_corner_windows(image_array)
    else:
        windows = hipp.core.define_center_window(image_array)

    template_array = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
    slices = hipp.core.slice_image_frame(image_array, windows)
    matches, _ = hipp.core.detect_fiducials(slices, template_array, windows)
    print(f"        → Coarse detection: {len(matches)} fiducials detected")

    locs, scores = hipp.core.detect_subpixel_fiducial_coordinates(
        image_file,
        image_array,
        matches,
        template_high_res_zoomed_file,
        labels=labels,
        distance_from_loc=distance_from_loc,
        factor=subpx_upsample_factor,
        qc=qc,
        qc_directory=qc_directory,
    )
    print(f"        → Subpixel refinement: {len(locs)} fiducials refined")
    return locs, scores


def iter_detect_fiducials(
    image_files_directory="input_data/raw_images/",
    image_file_name_column_name="fileName",
    image_files_extension=".tif",
    template_file=None,
    template_high_res_zoomed_file=None,
    midside_fiducials=False,
    corner_fiducials=False,
    center_fiducial=False,
    subpx_upsample_factor=8,
    qc=True,
    qc_directory="output_data/qc",
    n_workers=1,
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
    print(f"  Workers: {n_workers}")

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
    else:
        print(
            "Please specify midside or corner fiducials and provide corresponding templates."
        )
        return

    quality_score_labels = [s + "_score" for s in labels]
    template_array = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
    distance_from_loc = template_array.shape[0]

    total = len(images)
    args_list = [
        (
            image_file,
            template_file,
            template_high_res_zoomed_file,
            midside_fiducials,
            corner_fiducials,
            center_fiducial,
            labels,
            distance_from_loc,
            subpx_upsample_factor,
            qc,
            qc_directory,
            idx,
            total,
        )
        for idx, image_file in enumerate(images, 1)
    ]

    if n_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_detect_single_image, args_list))
    else:
        results = [_detect_single_image(a) for a in args_list]

    fiducial_locations = [r[0] for r in results]
    quality_scores = [r[1] for r in results]

    images_df = pd.DataFrame(images, columns=[image_file_name_column_name])
    fiducial_locations_df = pd.DataFrame(fiducial_locations, columns=labels)
    quality_scores_df = pd.DataFrame(quality_scores, columns=quality_score_labels)

    if center_fiducial:
        df = pd.concat([images_df, fiducial_locations_df, quality_scores_df], axis=1)
    else:
        principal_points_df = hipp.core.compute_principal_points(
            fiducial_locations_df, quality_scores_df
        )
        df = pd.concat(
            [images_df, fiducial_locations_df, quality_scores_df, principal_points_df],
            axis=1,
        )

    print(f"\n{'=' * 70}")
    print("[iter_detect_fiducials] Detection complete")
    print(f"  Total images processed: {total}")
    print(f"  Output DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"{'=' * 70}\n")

    return df


def _restitute_single_image(
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
            result["mid    side_angle_diff_before_tform"] = hipp.qc.compute_angle_diff(
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
        fid_valid = fiducial_coordinates[valid]
        fid_true_valid = fiducial_coordinates_true[valid]

        if len(fid_valid) >= 3:
            # tform = tf.AffineTransform() # Deprecated approach
            # tform.estimate(fid_valid, fid_true_valid)
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
    qc_directory="input_data/restitution_qc",
    n_workers=1,
):
    """
    Computes affine transformation between detected coordinates and true coordinates,
    then transforms image array.

    Parameters
    ----------
    df_detected : pd.DataFrame
        DataFrame with detected fiducial coordinates and image file names.
    fiducial_coordinates_true_mm : list or array-like, optional
        True fiducial coordinates in mm (x, y pairs, y positive upward).
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
        Whether to generate QC metrics, plots, and logs.
    keep_color : bool, default=False
        If False, convert images to grayscale. If True, keep all RGB channels.
    n_workers : int, default=1
        Number of parallel threads. tf.warp releases the GIL so threading is
        effective, but each worker holds a full float64 image array in RAM.
        Use n_workers=2 only if you have ample RAM (≥2× image size per worker).

    Returns
    -------
    tuple (qc_df, processing_log) if qc=True, else None.
        qc_df is also saved to <output_directory>/restitution_qc.csv.
        processing_log is saved to <output_directory>/restitution_log.json.

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
    if transform_image or crop_image:
        Path(output_directory).mkdir(parents=True, exist_ok=True)

    # ── Convert true coordinates once for the log ─────────────────────────────
    fiducial_coordinates_true_mm_arr = (
        np.array(fiducial_coordinates_true_mm, dtype=float)
        if fiducial_coordinates_true_mm is not None
        else None
    )

    # ── Build per-image kwargs ────────────────────────────────────────────────
    df_coords = df_detected.drop(
        [image_file_name_column_name, "principal_point_x", "principal_point_y"], axis=1
    )
    y_cols = [c for c in df_coords.columns if c.endswith("_y")]

    kwargs_list = []
    for index, row in df_coords.iterrows():
        fiducial_coordinates = np.array([[row[c[:-2] + "_x"], row[c]] for c in y_cols])
        principal_point = np.array([
            df_detected["principal_point_x"].loc[index],
            df_detected["principal_point_y"].loc[index],
        ])
        image_file = df_detected[image_file_name_column_name].loc[index]

        kwargs_list.append(
            dict(
                image_file=image_file,
                fiducial_coordinates=fiducial_coordinates,
                principal_point=principal_point,
                fiducial_coordinates_true_mm=fiducial_coordinates_true_mm_arr,
                transform_coords=transform_coords,
                transform_image=transform_image,
                crop_image=crop_image,
                keep_color=keep_color,
                qc=qc,
                interpolation_order=interpolation_order,
                image_square_dim=image_square_dim,
                output_directory=output_directory,
                scanning_resolution_mm=scanning_resolution_mm,
            )
        )

    # ── Dispatch ──────────────────────────────────────────────────────────────
    # ThreadPoolExecutor: tf.warp releases the GIL → genuine parallelism.
    # Keep n_workers low (default 1) because each active warp holds
    # a float64 copy of the image (~8× the uint8 size) in RAM.
    if n_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_restitute_single_image, **kw) for kw in kwargs_list
            ]
            results = [f.result() for f in futures]
    else:
        results = [_restitute_single_image(**kw) for kw in kwargs_list]

    if not qc:
        # Build and return the fiducial crop positions even without full QC.
        fiducials_in_crop = {
            Path(r["source_image"]).stem: r["fiducials_in_crop_px"]
            for r in results
            if r.get("source_image") and r.get("fiducials_in_crop_px") is not None
        }
        return fiducials_in_crop

    # ── Build QC DataFrame ────────────────────────────────────────────────────
    qc_metric_cols = [
        "coordinates_rmse_before_tform_mm",
        "coordinates_pp_dist_rmse_before_tform_mm",
        "midside_angle_diff_before_tform_mm",
        "corner_angle_diff_before_tform",
        "coordinates_rmse_after_tform",
        "coordinates_pp_dist_rmse_after_tform",
        "midside_angle_diff_after_tform",
        "corner_angle_diff_after_tform",
        "pixel_pitch_after_tform_x",
        "pixel_pitch_after_tform_y",
    ]
    qc_df = pd.DataFrame([{k: r[k] for k in qc_metric_cols} for r in results])
    qc_df.insert(
        0,
        image_file_name_column_name,
        df_detected[image_file_name_column_name].values,
    )
    qc_df.index = qc_df[image_file_name_column_name].str[-12:-4]

    # ── Save QC CSV ───────────────────────────────────────────────────────────
    qc_csv_path = os.path.join(output_directory, "restitution_qc.csv")
    qc_df.to_csv(qc_csv_path)
    print(f"QC data saved to {qc_csv_path}")

    # ── Build and save processing log ─────────────────────────────────────────
    def _jsonable(v):
        """Convert numpy scalars / NaN to JSON-serializable types."""
        if isinstance(v, float) and np.isnan(v):
            return None
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    log_results = [
        {
            k: _jsonable(v) if not isinstance(v, (dict, list)) else v
            for k, v in r.items()
        }
        for r in results
    ]
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
                fiducial_coordinates_true_mm_arr.tolist()
                if fiducial_coordinates_true_mm_arr is not None
                else None
            ),
        },
        "images": log_results,
    }
    log_path = os.path.join(output_directory, "restitution_log.json")
    with open(log_path, "w") as f:
        json.dump(processing_log, f, indent=2)
    print(f"Processing log saved to {log_path}")

    # ── QC plots ──────────────────────────────────────────────────────────────
    hipp.plot.plot_restitution_qc(qc_df, output_directory=qc_directory)

    # ── Build per-image fiducial positions in crop ────────────────────────────
    fiducials_in_crop = {
        Path(r["source_image"]).stem: r["fiducials_in_crop_px"]
        for r in results
        if r.get("source_image") and r.get("fiducials_in_crop_px") is not None
    }

    return qc_df, processing_log, fiducials_in_crop


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
                msg = "\n".join([
                    "WARNING: Irregular final image dimensions detected",
                    "Likely due to missing fiducial marker on a side of the images.",
                    "Check qc plots for:",
                ])
                print(msg)
                for i in images_tmp:
                    print(i)
                if not missing_proxy:
                    msg = "\n".join([
                        "Consider reprocessing by setting missing_proxy option to left, top, right, or bottom",
                        "to improve principal point detection.",
                    ])
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
            msg = "\n".join([
                "WARNING: Irregular final image dimensions detected",
                "Likely due to missing fiducial marker on a side of the images.",
                "Check qc plots for:",
            ])
            print(msg)
            for i in images:
                print(i)
            if not missing_proxy:
                msg = "\n".join([
                    "Consider reprocessing by setting missing_proxy option to left, top, right, or bottom",
                    "to improve principal point detection.",
                ])
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
            max_workers=max_workers,
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
                max_workers=max_workers,
            )
    if qc_df:
        print("Saving proxy detection QC dataframes to", qc_df_output_directory)
        p = Path(qc_df_output_directory)
        p.mkdir(parents=True, exist_ok=True)

        # Save to pickle files
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

        # Save also to CSV for easier human readability
        detected_df.to_csv(
            os.path.join(qc_df_output_directory, "detected_df.csv"), index=False
        )
        proxy_locations_df.to_csv(
            os.path.join(qc_df_output_directory, "proxy_locations_df.csv"), index=False
        )
        pd.DataFrame(principal_points, columns=["x", "y"]).to_csv(
            os.path.join(qc_df_output_directory, "principal_points.csv"), index=False
        )
        pd.DataFrame(distances, columns=["distance"]).to_csv(
            os.path.join(qc_df_output_directory, "distances.csv"), index=False
        )
        pd.DataFrame(intersection_angles, columns=["angle"]).to_csv(
            os.path.join(qc_df_output_directory, "intersection_angles.csv"), index=False
        )

    return image_square_dim
