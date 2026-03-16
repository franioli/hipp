import multiprocessing
import os
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import psutil

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
):

    locations_no_buffer = proxy_locations_df.iloc[:, 1:] - buffer_distance
    locations_no_buffer = locations_no_buffer.values.tolist()
    principal_points_no_buffer = np.array(principal_points) - buffer_distance

    pool = multiprocessing.Pool(processes=psutil.cpu_count(logical=False))
    for i in zip(images, locations_no_buffer, principal_points_no_buffer):
        pool.apply_async(hipp.plot.plot_proxies, args=(i, output_directory))
    pool.close()
    pool.join()


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


def plot_restitution_qc(qc_df):

    output_directory = "qc/restitution/"
    print("Image restitution qc plots in " + output_directory)
    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)

    y_labels = ["mm", "mm", "degree", "degree"]

    titles = [
        "Coordinates RMSE",
        "Coordinates distance to Principal Point RMSE",
        "Midside fiducial intersection angle at Principal Point difference",
        "Corner fiducial intersection angle at Principal Point difference",
    ]

    legend_labels = [
        "before transform",
        "after transform",
        "before transform",
        "after transform",
        "before transform",
        "after transform",
        "before transform",
        "after transform",
    ]

    output_names = [
        "coordinates_rmse",
        "coordinates_pp_dist_rmse",
        "midside_angle_diff",
        "corner_angle_diff",
    ]

    for i in np.arange(1, 5):
        fig, ax = plt.subplots(figsize=(12, 5))
        key1 = qc_df.iloc[:, i].name
        key2 = qc_df.iloc[:, i + 4].name
        qc_df[[key1, key2]].plot(ax=ax)
        ax.legend((legend_labels.pop(0), legend_labels.pop(0)))
        ax.xaxis.set_tick_params(rotation=90)
        ax.set_xlabel("")
        ax.set_ylabel(y_labels.pop(0))
        ax.set_title(titles.pop(0))

        out = os.path.join(output_directory, output_names.pop(0) + ".png")
        plt.savefig(out)
        # plt.close()


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
    """
    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)

    full_res_dir = os.path.join(output_directory, "full_res")
    if save_full_res:
        pathlib.Path(full_res_dir).mkdir(parents=True, exist_ok=True)

    df_cols = df_detected.columns.tolist()
    n = len(df_detected)
    grid_rows = int(np.ceil(n / grid_cols))

    if save_grid:
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
        axes = np.array(axes).flatten()

    for idx, (_, row) in enumerate(df_detected.iterrows()):
        image_file = str(row[image_file_name_column])
        coords = _coords_from_row(row, df_cols, image_file_name_column)
        _, name, ext = hipp.io.split_file(image_file)

        if verbose:
            print(f"  [{idx + 1}/{n}] {name}{ext}  —  {len(coords)} fiducials")

        # ── Full-resolution OpenCV output ─────────────────────────────────────
        if save_full_res:
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  WARNING: cannot read {image_file}")
            else:
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

        # ── Matplotlib thumbnail ──────────────────────────────────────────────
        if save_grid:
            ax = axes[idx]
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                nh = max(1, int(h * scale_factor))
                nw = max(1, int(w * scale_factor))
                thumb = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
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

    if save_grid:
        for i in range(idx + 1, len(axes)):
            axes[i].axis("off")

        from matplotlib.lines import Line2D

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
