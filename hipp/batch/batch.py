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
from tqdm.auto import tqdm

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


def _detect_single_image(
    image_file: str | Path,
    template_file: str | Path,
    template_high_res_zoomed_file: str | None,
    midside_fiducials: bool = False,
    corner_fiducials: bool = False,
    center_fiducial: bool = False,
    labels: list[str] | None = None,
    distance_from_loc: int = 100,
    subpx_upsample_factor: int = 8,
    qc: bool = True,
    qc_directory: str = "output_data/qc",
) -> tuple[list[tuple[float, float]], list[float]]:
    """Detect fiducials for a single image and refine them to subpixel precision.

    Loads the image, defines the detection windows based on the requested fiducial
    geometry, performs coarse template matching, and refines each detected
    fiducial using the high-resolution template.

    Args:
        image_file: Path to the image being processed.
        template_file: Path to the low-resolution template used for coarse matching.
        template_high_res_zoomed_file: Path to the high-resolution template used
            for subpixel refinement.
        midside_fiducials: Whether to detect midside fiducials (left, top, right, bottom).
        corner_fiducials: Whether to detect corner fiducials (top-left, top-right, bottom-right, bottom-left).
        center_fiducial: Whether to detect the center fiducial (principal point).
        labels: Fiducial labels corresponding to the expected match order (e.g. ["midside_left", "midside_top", "midside_right", "midside_bottom"]).
        distance_from_loc: Crop half-size in pixels used around each coarse match before subpixel refinement (should be large enough to contain the full high-res template).
        subpx_upsample_factor: Upsampling factor for subpixel refinement (i.e. how many subpixels per original pixel, typically 8 or 16).
        qc: Whether to write QC outputs during high-resolution matching.
        qc_directory: Directory where QC outputs are written.

    Returns:
        A tuple ``(locs, scores)`` where ``locs`` contains the refined fiducial
        coordinates as ``(y, x)`` pairs and ``scores`` contains the corresponding
        matching quality scores.
    """
    image_array = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)

    if midside_fiducials:
        windows = hipp.core.define_midside_windows(image_array)
    elif corner_fiducials:
        windows = hipp.core.define_corner_windows(image_array)
    elif center_fiducial:
        windows = hipp.core.define_center_window(image_array)
    else:
        raise ValueError(
            "At least one fiducial type must be specified: midside, corner, or center."
        )

    template_array = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
    slices = hipp.core.slice_image_frame(image_array, windows)
    matches, _ = hipp.core.detect_fiducials(slices, template_array, windows)
    print(f"        → Coarse detection: {len(matches)} fiducials detected")

    locs, scores = hipp.core.detect_subpixel_fiducial_coordinates(
        image_file,
        image_array,
        matches,
        str(template_high_res_zoomed_file),
        labels=labels,
        distance_from_loc=distance_from_loc,
        factor=subpx_upsample_factor,
        qc=qc,
        qc_directory=qc_directory,
    )
    print(f"        → Subpixel refinement: {len(locs)} fiducials refined")

    return locs, scores


def iter_detect_fiducials(
    image_files_directory: str = "input_data/raw_images/",
    image_file_name_column_name: str = "fileName",
    image_files_extension: str = ".tif",
    template_file: str | None = None,
    template_high_res_zoomed_file: str | None = None,
    midside_fiducials: bool = False,
    corner_fiducials: bool = False,
    center_fiducial: bool = False,
    subpx_upsample_factor: int = 8,
    qc: bool = True,
    qc_directory: str = "output_data/qc",
    n_workers: int = 1,
) -> pd.DataFrame | None:
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
    if template_high_res_zoomed_file is not None:
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

    if template_file is None:
        raise ValueError("template_file must be provided for fiducial detection.")
    if template_high_res_zoomed_file is None:
        raise ValueError(
            "template_high_res_zoomed_file must be provided for fiducial detection."
        )

    template_array = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
    if template_array is None:
        raise ValueError(f"Could not read template image: {template_file}")
    distance_from_loc = template_array.shape[0]

    total = len(images)
    kwargs_list = [
        {
            "image_file": image_file,
            "template_file": template_file,
            "template_high_res_zoomed_file": template_high_res_zoomed_file,
            "midside_fiducials": midside_fiducials,
            "corner_fiducials": corner_fiducials,
            "center_fiducial": center_fiducial,
            "labels": labels,
            "distance_from_loc": distance_from_loc,
            "subpx_upsample_factor": subpx_upsample_factor,
            "qc": qc,
            "qc_directory": qc_directory,
        }
        for image_file in images
    ]

    if n_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_detect_single_image, **kwargs)
                for kwargs in kwargs_list
            ]
            results_by_index = {}
            future_to_index = {future: index for index, future in enumerate(futures)}
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=total,
                desc="Detecting fiducials",
            ):
                index = future_to_index[future]
                results_by_index[index] = future.result()
            results = [results_by_index[index] for index in range(total)]
    else:
        results = [
            _detect_single_image(**kwargs)
            for kwargs in tqdm(
                kwargs_list,
                total=total,
                desc="Detecting fiducials",
            )
        ]

    fiducial_locations = [r[0] for r in results]
    quality_scores = [r[1] for r in results]

    images_df = pd.DataFrame(images, columns=[image_file_name_column_name])
    fiducial_locations_df = pd.DataFrame(fiducial_locations, columns=labels)
    quality_score_labels = [s + "_score" for s in labels]
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
    geometric_outlier_threshold_px=200.0,
):
    """Computes affine transformation between detected coordinates and true coordinates,
    then transforms image array.

            results_by_index = {}
    ----------
    df_detected : pd.DataFrame
        DataFrame with detected fiducial coordinates and image file names.
    fiducial_coordinates_true_mm : list or array-like, optional
        True fiducial coordinates in mm (x, y pairs, y positive upward).
    image_file_name_column_name : str, default="fileName"
        Column name containing image file paths.
    scanning_resolution_mm : float, default=0.02
        Scanning resolution in mm/px.
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
                geometric_outlier_threshold_px=geometric_outlier_threshold_px,
            )
        )

    # ── Dispatch ──────────────────────────────────────────────────────────────
    # ThreadPoolExecutor: tf.warp releases the GIL → genuine parallelism.
    # Keep n_workers low (default 1) because each active warp holds
    # a float64 copy of the image (~8× the uint8 size) in RAM.
    if n_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(hipp.core.restitute_single_image, **kw)
                for kw in kwargs_list
            ]
            results = [f.result() for f in futures]
    else:
        results = [hipp.core.restitute_single_image(**kw) for kw in kwargs_list]

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

    # Keep backward-compatible mm-suffixed columns in CSV while exposing
    # canonical *_before_tform names expected by plot_restitution_qc.
    qc_df["coordinates_rmse_before_tform"] = qc_df["coordinates_rmse_before_tform_mm"]
    qc_df["coordinates_pp_dist_rmse_before_tform"] = qc_df[
        "coordinates_pp_dist_rmse_before_tform_mm"
    ]
    qc_df["midside_angle_diff_before_tform"] = qc_df[
        "midside_angle_diff_before_tform_mm"
    ]

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
