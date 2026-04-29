import os
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from matplotlib.lines import Line2D

import hipp.io
import hipp.plot

"""
Library for common plotting functions.
"""


def iter_plot_proxies(
    images,
    proxy_locations_df,
    principal_points,
    buffer_distance=250,
    output_directory="qc/proxy_detection",
    verbose=True,
    max_workers=None,
):

    locations_no_buffer = proxy_locations_df.iloc[:, 1:] - buffer_distance
    locations_no_buffer = locations_no_buffer.values.tolist()
    principal_points_no_buffer = np.array(principal_points) - buffer_distance

    max_workers = max_workers or psutil.cpu_count(logical=False) or 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(hipp.plot.plot_proxies, data, output_directory)
            for data in zip(
                images, locations_no_buffer, principal_points_no_buffer, strict=True
            )
        ]
        for future in as_completed(futures):
            future.result()


def plot_histogram(image_array, figsize=(10, 5)):

    fig, ax = plt.subplots(figsize=figsize)

    n, bins, patches = ax.hist(image_array.ravel()[::40], bins=256, range=(0, 256))
    plt.show()


def plot_images(
    image_arrays,
    rows=5,
    columns=5,
    figsize=(10, 10),
    cmap="gray",
    title=None,
    labels=None,
    output_file_name=None,
):

    plt.figure(figsize=figsize)

    for i in range(rows * columns):
        ax = plt.subplot(rows, columns, i + 1)

        try:
            image = image_arrays[i]
            ax.imshow(image, cmap=cmap)
            ax.set_xticks(())
            ax.set_yticks(())

            if isinstance(labels, type(list())):
                ax.set_title(labels[i])
        except:
            ax.axis("off")
            pass

    if isinstance(title, str):
        plt.suptitle(title, fontsize=15)
        plt.subplots_adjust(top=0.95)

    plt.tight_layout()

    if isinstance(output_file_name, str):
        file_path, file_name, file_extension = hipp.io.split_file(output_file_name)

        p = pathlib.Path(file_path)
        p.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_file_name)


def plot_restitution_qc(qc_df, output_directory="qc/restitution/"):

    print("Image restitution qc plots in " + output_directory)
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    n_images = len(qc_df)
    x = np.arange(n_images)
    image_labels = qc_df.index.tolist()  # truncated names set by caller

    # ── Save index → image name mapping for reference ─────────────────────────
    pd.DataFrame({"index": x, "image": image_labels}).to_csv(
        os.path.join(output_directory, "image_index_map.csv"), index=False
    )

    # ── Match before/after column pairs by name ───────────────────────────────
    before_cols = [c for c in qc_df.columns if c.endswith("_before_tform")]
    pairs = []
    for bc in before_cols:
        ac = bc.replace("_before_tform", "_after_tform")
        if ac in qc_df.columns:
            pairs.append((bc, ac))

    titles = {
        "coordinates_rmse": ("Coordinates RMSE", "mm"),
        "coordinates_pp_dist_rmse": (
            "Coordinates distance to Principal Point RMSE",
            "mm",
        ),
        "midside_angle_diff": (
            "Midside fiducial intersection angle difference",
            "degree",
        ),
        "corner_angle_diff": (
            "Corner fiducial intersection angle difference",
            "degree",
        ),
    }

    # Figure width: enough so labels don't overlap (min 12, ~0.5 per image)
    fig_width = max(12, n_images * 0.5)

    for before_col, after_col in pairs:
        # derive metric key from column name
        metric = before_col.replace("_before_tform", "")
        title, y_label = titles.get(metric, (metric, ""))

        fig, ax = plt.subplots(figsize=(fig_width, 5))
        ax.plot(
            x,
            qc_df[before_col].values,
            marker="o",
            markersize=4,
            label="before transform",
        )
        ax.plot(
            x,
            qc_df[after_col].values,
            marker="o",
            markersize=4,
            label="after transform",
        )

        # ── X-axis labels ──────────────────────────────────────────────────────
        if n_images <= 40:
            # Show truncated name for every image, rotated
            ax.set_xticks(x)
            fontsize = max(5, 9 - n_images // 8)
            ax.set_xticklabels(image_labels, rotation=90, fontsize=fontsize)
        else:
            # Too many images: show integer index; a stride of ~20 tick marks
            step = max(1, n_images // 20)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(x[::step])
            ax.set_xlabel("Image index  (see image_index_map.csv)")

        ax.legend()
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        out = os.path.join(output_directory, metric + ".png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_proxies(data, output_directory=None):

    image_file = data[0]
    proxies = np.array(data[1])
    proxies_x = proxies[1::2]
    proxies_y = proxies[::2]
    principal_point = data[2]
    principal_point_x = principal_point[1]
    principal_point_y = principal_point[0]

    if isinstance(output_directory, type(None)):
        output_directory = "qc/proxy_detection"

    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)

    path, name, ext = hipp.io.split_file(image_file)

    image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_array, cmap="gray")
    ax.scatter(proxies_x, proxies_y, color="lime", marker=".")
    ax.scatter(principal_point_x, principal_point_y, color="red", marker=".")
    plt.tight_layout()

    output_file_name = os.path.join(output_directory, name + ".png")

    fig.savefig(output_file_name)
    plt.close(fig)
    return output_file_name


# ── Detected fiducials visualisation ────────────────────────────────────────


def _color_for_label(label, bgr=True):
    """Return a color tuple for a fiducial label (BGR for OpenCV, RGB for matplotlib)."""
    if "principal" in label:
        return (0, 0, 255) if bgr else (1.0, 0.0, 0.0)  # red
    if "corner" in label:
        return (255, 140, 0) if bgr else (1.0, 0.55, 0.0)  # orange
    if "midside" in label:
        return (0, 220, 0) if bgr else (0.0, 0.86, 0.0)  # lime-green
    return (0, 255, 255) if bgr else (0.0, 1.0, 1.0)  # yellow (fallback)


def _coords_from_row(row, df_columns, skip_col):
    """
    Extract {label: (y, x)} from one DataFrame row.
    Handles both tuple-value columns (from iter_detect_fiducials)
    and split _y/_x columns (after split_position_tuples / eval_matches).
    """
    non_score = [c for c in df_columns if c != skip_col and not c.endswith("_score")]

    y_cols = [c for c in non_score if c.endswith("_y")]
    coords = {}

    if y_cols:  # split format  (label_y / label_x)
        for yc in y_cols:
            label = yc[:-2]
            xc = label + "_x"
            if xc not in df_columns:
                continue
            y, x = row.get(yc, np.nan), row.get(xc, np.nan)
            try:
                if not (np.isnan(float(y)) or np.isnan(float(x))):
                    coords[label] = (float(y), float(x))
            except (TypeError, ValueError):
                pass
    else:  # tuple format  (y, x) in each cell
        for col in non_score:
            val = row.get(col)
            if val is None:
                continue
            if isinstance(val, (tuple, list, np.ndarray)) and len(val) == 2:
                try:
                    y, x = float(val[0]), float(val[1])
                    if not (np.isnan(y) or np.isnan(x)):
                        coords[col] = (y, x)
                except (TypeError, ValueError):
                    pass

    return coords


def _draw_full_res_single(args):
    """
    Worker: draw fiducials on one full-resolution image and save it.
    Returns (idx, name+ext, out_path) on success or (idx, name+ext, error_msg) on failure.
    """
    image_file, coords, full_res_dir, marker_radius, draw_crosshair = args
    _, name, ext = hipp.io.split_file(image_file)
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (name + ext, None, f"WARNING: cannot read {image_file}")

    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    thickness = max(2, marker_radius // 6)
    crosshair_len = marker_radius * 3

    for label, (y, x) in coords.items():
        cx, cy = int(round(x)), int(round(y))
        color = _color_for_label(label, bgr=True)
        cv2.circle(canvas, (cx, cy), marker_radius, color, thickness)
        if draw_crosshair:
            cv2.line(
                canvas,
                (cx - crosshair_len, cy),
                (cx + crosshair_len, cy),
                color,
                max(1, thickness // 2),
            )
            cv2.line(
                canvas,
                (cx, cy - crosshair_len),
                (cx, cy + crosshair_len),
                color,
                max(1, thickness // 2),
            )
        cv2.putText(
            canvas,
            label,
            (cx + marker_radius + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )

    out_path = os.path.join(full_res_dir, name + "_fiducials" + ext)
    cv2.imwrite(out_path, canvas)
    return (name + ext, out_path, None)


def _load_thumbnail(args):
    """
    Worker: read one image, downscale it, and return (idx, thumb, coords).
    Returns None for the thumb on read failure.
    """
    idx, image_file, coords, scale_factor = args
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (idx, None, coords)
    h, w = img.shape
    nh = max(1, int(h * scale_factor))
    nw = max(1, int(w * scale_factor))
    thumb = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return (idx, thumb, coords)


def iter_plot_detected_fiducials(
    df_detected,
    image_file_name_column="fileName",
    output_directory="qc/fiducial_detection",
    marker_radius=30,
    draw_crosshair=True,
    save_full_res=True,
    scale_factor=0.1,
    grid_cols=4,
    figsize=(20, 20),
    save_grid=True,
    verbose=True,
    n_workers=None,
):
    """
    Visualise detected fiducial markers on images.

    Two QC outputs:

    1. **Full-resolution OpenCV images** (one per input image) saved to
       ``<output_directory>/full_res/``.  All fiducial marks are drawn as
       coloured circles + crosshairs on top of the original image.
       Suitable for pixel-level accuracy inspection.

    2. **Matplotlib overview grid** (one figure for the whole batch) saved to
       ``<output_directory>/overview_grid.png``.  Each thumbnail shows the
       image at ``scale_factor`` of its original size with markers overlaid.
       Suitable for a quick batch sanity-check.

    Marker colours: orange = corner, lime-green = midside, red = principal point.

    Parameters
    ----------
    df_detected : pd.DataFrame
        Output of ``hipp.batch.iter_detect_fiducials`` (tuple columns)
        or ``hipp.core.eval_matches(split_position_tuples=True)`` (_y/_x columns).
    image_file_name_column : str
        Column that contains the image file paths.
    output_directory : str
        Root directory for QC outputs.
    marker_radius : int
        Radius in pixels of circles on full-resolution images.
    draw_crosshair : bool
        Draw crosshair lines through each circle on full-resolution images.
    save_full_res : bool
        Save annotated full-resolution images via OpenCV.
    scale_factor : float
        Downscale fraction for the matplotlib grid thumbnails (0.1 = 10 %).
    grid_cols : int
        Number of columns in the overview grid.
    figsize : tuple
        Figure size passed to ``plt.subplots``.
    save_grid : bool
        Save the matplotlib overview grid.
    verbose : bool
        Print progress to stdout.
    n_workers : int
        Number of threads for parallel I/O (full-res writing and thumbnail loading). Defaults to the number of physical CPU cores.
    """
    if n_workers is None:
        n_workers = psutil.cpu_count(logical=False) or 1

    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)

    full_res_dir = os.path.join(output_directory, "full_res")
    if save_full_res:
        pathlib.Path(full_res_dir).mkdir(parents=True, exist_ok=True)

    df_cols = df_detected.columns.tolist()
    n = len(df_detected)
    grid_rows = int(np.ceil(n / grid_cols))

    # Pre-build per-row data once (avoid repeated iterrows inside workers)
    rows_data = [
        (
            str(row[image_file_name_column]),
            _coords_from_row(row, df_cols, image_file_name_column),
        )
        for _, row in df_detected.iterrows()
    ]

    # ── Matplotlib overview grid — thumbnails loaded in parallel ──────────────
    if save_grid:
        thumb_args = [
            (idx, image_file, coords, scale_factor)
            for idx, (image_file, coords) in enumerate(rows_data)
        ]

        # Load / downscale all thumbnails concurrently; collect in index order
        thumbnails: dict[int, tuple] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for idx, thumb, coords in executor.map(_load_thumbnail, thumb_args):
                thumbnails[idx] = (thumb, coords)

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
        axes = np.array(axes).flatten()

        for idx in range(n):
            image_file, coords = rows_data[idx]
            thumb, _ = thumbnails[idx]
            _, name, ext = hipp.io.split_file(image_file)
            ax = axes[idx]
            if thumb is not None:
                ax.imshow(thumb, cmap="gray", origin="upper")
                for label, (y, x) in coords.items():
                    mpl_color = _color_for_label(label, bgr=False)
                    marker = "x" if "principal" in label else "+"
                    ax.scatter(
                        [x * scale_factor],
                        [y * scale_factor],
                        c=[mpl_color],
                        s=80,
                        marker=marker,
                        linewidths=2,
                        zorder=5,
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "load error",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
            ax.set_title(name, fontsize=6, pad=2)
            ax.axis("off")

        for i in range(n, len(axes)):
            axes[i].axis("off")

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="+",
                linestyle="None",
                markersize=9,
                markeredgewidth=2,
                color="orange",
                label="corner",
            ),
            Line2D(
                [0],
                [0],
                marker="+",
                linestyle="None",
                markersize=9,
                markeredgewidth=2,
                color="limegreen",
                label="midside",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="None",
                markersize=9,
                markeredgewidth=2,
                color="red",
                label="principal point",
            ),
        ]
        fig.legend(handles=legend_handles, loc="lower right", fontsize=10)
        plt.tight_layout(pad=0.5)
        grid_out = os.path.join(output_directory, "fiducials_overview_grid.png")
        plt.savefig(grid_out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"\nOverview grid  → {grid_out}")

    # ── Full-resolution OpenCV output — parallel ──────────────────────────────
    if save_full_res:
        full_res_args = [
            (image_file, coords, full_res_dir, marker_radius, draw_crosshair)
            for image_file, coords in rows_data
        ]
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = list(
                executor.submit(_draw_full_res_single, a) for a in full_res_args
            )
            for completed, future in enumerate(as_completed(futures), 1):
                name_ext, out_path, err = future.result()
                if err:
                    print(f"  [{completed}/{n}] {err}")
                elif verbose:
                    print(f"  [{completed}/{n}] {name_ext}  →  {out_path}")

    if save_full_res and verbose:
        print(f"Full-res images → {full_res_dir}/")


## some helper functions
def check_if_number_even(n):
    """
    checks if int n is an even number
    """
    if (n % 2) == 0:
        return True
    else:
        return False


def make_number_even(n):
    """
    adds 1 to int n if odd number
    """
    if check_if_number_even(n):
        return n
    else:
        return n + 1


def get_row_column(n):
    """
    returns largest factor pair for int n
    makes rows the larger number
    """
    max_pair = max([(i, n / i) for i in range(1, int(n**0.5) + 1) if n % i == 0])
    rows = int(max(max_pair))
    columns = int(min(max_pair))

    # in case n is odd
    # check if you get a smaller pair by adding 1 to make number even
    if not check_if_number_even(n):
        n = make_number_even(n)
        max_pair = max([(i, n / i) for i in range(1, int(n**0.5) + 1) if n % i == 0])
        alt_rows = int(max(max_pair))
        alt_columns = int(min(max_pair))

        if (rows, columns) > (alt_rows, alt_columns):
            return (alt_rows, alt_columns)
        else:
            return (rows, columns)
    return (rows, columns)
