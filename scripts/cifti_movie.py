"""Render a CIFTI-2 dense timeseries (dtseries) to frames and optionally encode an MP4.

This is a sibling to scripts/nifti_movie.py, but for surface-based CIFTI files in
fsLR32k space.

- Input: dense CIFTI dtseries (.dtseries.nii) or scalar series (.dscalar.nii)
- Optional atlas: dense label CIFTI (.dlabel.nii)
- Surface rendering: fsLR32k inflated GIFTI surfaces

Video encoding uses an ffmpeg executable. The script will try, in order:
1) --ffmpeg path (if provided)
2) system ffmpeg on PATH
3) imageio-ffmpeg (if installed; see optional dependency extra `viz`)

Examples
--------
# Basic movie
uv run python scripts/cifti_movie.py --input sub-01_task-rest.dtseries.nii --output bold.mp4

# With an atlas stats subplot (enabled automatically when --atlas is passed)
uv run python scripts/cifti_movie.py --input sub-01_task-rest.dtseries.nii --atlas TY7.dlabel.nii \
  --output bold_stats.mp4 --atlas-stats-max-points-per-label 300

# Multiple surface views (each view shows both hemispheres)
uv run python scripts/cifti_movie.py --input sub-01_task-rest.dtseries.nii --output bold_2views.mp4 \
  --surf-views lateral medial
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, cast

import matplotlib

# Force a non-interactive backend for speed and headless runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import nibabel as nib
from nibabel.loadsave import load as nib_load
import numpy as np
from nilearn import plotting, surface
from nilearn.plotting.cm import mix_colormaps


@dataclasses.dataclass(frozen=True)
class _CiftiFrameSpec:
    frame_axis_first: bool
    n_frames: int


def _apply_surf_zoom(ax: Axes, zoom: float) -> None:
    """Zoom mplot3d axes in a version-tolerant way.

    Matplotlib's 3D API differs by version; using `Axes3D.dist` is broadly
    supported and reduces the viewing distance (smaller = closer).
    """

    try:
        z = float(zoom)
    except Exception:
        return

    if not np.isfinite(z) or z <= 0:
        return

    # Matplotlib default dist is typically 10.
    try:
        base = 10.0
        new_dist = max(1.0, base / z)
        setattr(cast(Any, ax), "dist", new_dist)
    except Exception:
        return


def _apply_surf_view_offsets(
    ax: Axes, *, elev_offset: float, azim_offset: float
) -> None:
    """Adjust the 3D camera relative to nilearn's built-in `view=` presets.

    Nilearn sets a base (elev, azim) pair for common views like 'lateral'.
    Offsets let us tilt/rotate slightly (e.g., to better see into sulci)
    while keeping the same high-level view preset.
    """

    try:
        de = float(elev_offset)
        da = float(azim_offset)
    except Exception:
        return

    if (not np.isfinite(de)) or (not np.isfinite(da)):
        return
    if de == 0.0 and da == 0.0:
        return

    ax_any = cast(Any, ax)
    try:
        base_elev = float(getattr(ax_any, "elev", 0.0))
        base_azim = float(getattr(ax_any, "azim", 0.0))
        ax_any.view_init(elev=base_elev + de, azim=base_azim + da)
    except Exception:
        return


def _load_vertex_mask_gifti(path: Path) -> np.ndarray:
    """Load a per-vertex ROI mask from a GIFTI (.func.gii or .label.gii).

    Nonzero vertices are treated as True. Returns a boolean array (n_vertices,).
    """

    img = nib.load(str(path))  # type: ignore[assignment]
    darrays = getattr(img, "darrays", None)
    if darrays is None or len(darrays) == 0:
        raise ValueError(f"No data arrays found in ROI mask GIFTI: {path}")

    data = np.asarray(darrays[0].data).ravel()
    if data.ndim != 1:
        raise ValueError(f"Expected 1D ROI mask data in {path}, got shape {data.shape}")

    data_f = np.asarray(data, dtype=float)
    return np.isfinite(data_f) & (data_f != 0.0)


def _focus_3d_axes_on_mask(
    ax: Axes, *, coords: np.ndarray, mask: np.ndarray, pad_mm: float
) -> None:
    """Crop a 3D axes to the bounding box of masked vertices (+ padding)."""

    xyz = np.asarray(coords, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected coords shape (n_vertices, 3), got {xyz.shape}")

    m = np.asarray(mask, dtype=bool).ravel()
    if m.shape[0] != xyz.shape[0]:
        raise ValueError(
            f"ROI mask length ({m.shape[0]}) does not match surface vertices ({xyz.shape[0]})"
        )

    roi = xyz[m]
    if roi.size == 0:
        return

    pad = float(pad_mm)
    lo = roi.min(axis=0) - pad
    hi = roi.max(axis=0) + pad

    # Disable autoscaling (nilearn/mplot3d tends to keep it on).
    ax_any = cast(Any, ax)
    ax_any.set_autoscale_on(False)
    ax_any.set_xlim(float(lo[0]), float(hi[0]))
    ax_any.set_ylim(float(lo[1]), float(hi[1]))
    ax_any.set_zlim(float(lo[2]), float(hi[2]))


def _add_colorbar(
    *, fig: Figure, cax: Axes, cmap: str, vmin: float, vmax: float
) -> None:
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation="vertical")


def _compute_bg_facecolors(*, n_faces: int, alpha: float) -> np.ndarray:
    # With bg_map=None, nilearn uses bg_data=0.5 everywhere, which maps to a
    # mid-gray in gray_r. Keep the same look, but precompute once.
    bg = plt.get_cmap("gray_r")(np.full(n_faces, 0.5, dtype=float))
    bg[:, 3] = float(alpha) * bg[:, 3]
    return bg


def _update_poly3d_facecolors(
    *,
    poly: Any,
    faces: np.ndarray,
    stat_map_vertices: np.ndarray,
    cmap: Any,
    vmin: float,
    vmax: float,
    bg_facecolors: np.ndarray,
) -> None:
    # Matplotlib backend in nilearn computes face values from vertex values.
    face_vals = np.mean(np.asarray(stat_map_vertices, dtype=float)[faces], axis=1)
    kept = ~np.isnan(face_vals)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        finite = face_vals[np.isfinite(face_vals)]
        if finite.size:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
        else:
            vmin, vmax = -1.0, 1.0
        if vmin == vmax:
            vmax = vmin + 1e-6

    scaled = (face_vals - float(vmin)) / (float(vmax) - float(vmin))
    # Keep NaNs as NaNs; colormap will ignore via alpha=0 below.
    scaled = np.clip(scaled, 0.0, 1.0)

    surf_colors = cmap(scaled)
    surf_colors[~kept, 3] = 0.0

    face_colors = mix_colormaps(surf_colors, bg_facecolors)
    face_colors = np.clip(face_colors, 0.0, 1.0)

    poly.set_facecolors(face_colors)


def _find_ffmpeg(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path

    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _run_ffmpeg(
    *,
    ffmpeg: str,
    frames_pattern: str,
    fps: float,
    output: Path,
    crf: int,
    preset: str,
    scale: str | None,
) -> None:
    cmd: list[str] = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        frames_pattern,
    ]

    vf_filters: list[str] = []
    if scale:
        vf_filters.append(f"scale={scale}")

    # libx264 (and yuv420p) require even width/height.
    vf_filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")

    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]

    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        preset,
        str(output),
    ]

    subprocess.run(cmd, check=True)


def _iter_frame_indices(
    n_frames: int, start: int, stop: int | None, step: int
) -> Iterable[int]:
    stop_ = n_frames if stop is None else min(stop, n_frames)
    if start < 0 or start >= n_frames:
        raise ValueError(f"start must be in [0, {n_frames - 1}], got {start}")
    if stop_ <= start:
        raise ValueError(f"stop must be > start; got start={start} stop={stop_}")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    return range(start, stop_, step)


def _validate_percentiles(p_low: float, p_high: float) -> tuple[float, float]:
    if not (0.0 <= p_low <= 100.0 and 0.0 <= p_high <= 100.0):
        raise ValueError(f"Percentiles must be in [0, 100], got {p_low}, {p_high}")
    if p_low >= p_high:
        raise ValueError(
            f"Lower percentile must be < upper percentile, got {p_low}, {p_high}"
        )
    return float(p_low), float(p_high)


def _sample_finite_values(
    data: np.ndarray, *, max_samples: int, rng: np.random.Generator
) -> np.ndarray:
    flat = np.asarray(data).ravel()
    n = int(flat.size)
    if n == 0 or max_samples <= 0:
        return np.asarray([], dtype=float)

    if n <= max_samples:
        vals = flat[np.isfinite(flat)]
        return vals.astype(float, copy=False)

    collected: list[np.ndarray] = []
    remaining = max_samples
    for _ in range(25):
        if remaining <= 0:
            break
        draw = min(max(remaining * 2, 1024), 200_000)
        idx = rng.integers(0, n, size=draw, dtype=np.int64)
        vals = flat[idx]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if vals.size > remaining:
            vals = vals[:remaining]
        collected.append(vals.astype(float, copy=False))
        remaining -= int(vals.size)

    if not collected:
        return np.asarray([], dtype=float)
    return np.concatenate(collected, axis=0)


def _infer_cifti_axes(
    img: nib.cifti2.cifti2.Cifti2Image,
) -> tuple[Any, Any, _CiftiFrameSpec]:
    """Return (frame_axis, brain_axis, frame_spec).

    Uses nibabel.cifti2.cifti2_axes.from_index_mapping to remain compatible with
    the installed nibabel version.
    """

    from nibabel.cifti2 import cifti2_axes

    ax0 = cifti2_axes.from_index_mapping(img.header.get_index_map(0))
    ax1 = cifti2_axes.from_index_mapping(img.header.get_index_map(1))

    if isinstance(ax0, cifti2_axes.BrainModelAxis) and not isinstance(
        ax1, cifti2_axes.BrainModelAxis
    ):
        # brain x frames
        frame_axis, brain_axis = ax1, ax0
        frame_spec = _CiftiFrameSpec(
            frame_axis_first=False, n_frames=int(frame_axis.size)
        )
    elif isinstance(ax1, cifti2_axes.BrainModelAxis) and not isinstance(
        ax0, cifti2_axes.BrainModelAxis
    ):
        # frames x brain
        frame_axis, brain_axis = ax0, ax1
        frame_spec = _CiftiFrameSpec(
            frame_axis_first=True, n_frames=int(frame_axis.size)
        )
    else:
        raise ValueError(
            "Expected one BrainModelAxis and one frame axis (SeriesAxis/ScalarAxis). "
            f"Got axis0={type(ax0)} axis1={type(ax1)}"
        )

    return frame_axis, brain_axis, frame_spec


def _get_frame_vector(
    img: nib.cifti2.cifti2.Cifti2Image, *, frame_index: int, frame_spec: _CiftiFrameSpec
) -> np.ndarray:
    dataobj = img.dataobj
    if frame_spec.frame_axis_first:
        vec = np.asanyarray(dataobj[frame_index, :])
    else:
        vec = np.asanyarray(dataobj[:, frame_index])
    return np.asarray(vec).ravel().astype(float, copy=False)


def _extract_cortex_structures(brain_axis: Any) -> dict[str, tuple[slice, Any]]:
    """Return structure_name -> (slice, brain_model_for_structure)."""

    structures: dict[str, tuple[slice, Any]] = {}
    for struct_name, struct_slice, struct_bm in brain_axis.iter_structures():
        structures[str(struct_name)] = (struct_slice, struct_bm)
    return structures


def _brain_to_hemi_vertices(
    *,
    frame_vec: np.ndarray,
    structures: dict[str, tuple[slice, Any]],
    structure_name: str,
) -> np.ndarray:
    if structure_name not in structures:
        raise ValueError(f"Structure {structure_name} not found in CIFTI brain models")

    struct_slice, struct_bm = structures[structure_name]
    vals = np.asarray(frame_vec[struct_slice], dtype=float)

    # For cortex structures, the brain model provides vertex indices into a
    # full fsLR mesh (e.g., 32492 for fsLR32k). The slice contains only the
    # modeled vertices (medial wall already removed), so we must scatter the
    # values into a full-length vertex array.
    struct_bm_any = cast(Any, struct_bm)

    vertex = np.asarray(struct_bm_any.vertex, dtype=np.int64)
    nverts_dict = getattr(struct_bm_any, "nvertices", None)
    if isinstance(nverts_dict, dict) and structure_name in nverts_dict:
        n_verts = int(nverts_dict[structure_name])
    else:
        n_verts = int(vertex.max()) + 1 if vertex.size else int(vals.size)

    if int(vertex.size) != int(vals.size):
        raise ValueError(
            f"BrainModel vertex index length ({int(vertex.size)}) does not match values length ({int(vals.size)})"
        )

    out = np.full(n_verts, np.nan, dtype=float)
    out[vertex] = vals
    return out


def _compute_cortex_mask(
    brain_axis: Any, structures: dict[str, tuple[slice, Any]]
) -> np.ndarray:
    mask = np.zeros(int(brain_axis.size), dtype=bool)
    for key in (
        "CIFTI_STRUCTURE_CORTEX_LEFT",
        "CIFTI_STRUCTURE_CORTEX_RIGHT",
    ):
        if key in structures:
            sl, _bm = structures[key]
            mask[sl] = True
    return mask


def _prepare_label_indices_from_grayordinates(
    *,
    atlas_gray: np.ndarray,
    cortex_mask: np.ndarray,
    ignore_zero: bool,
) -> tuple[list[int], list[np.ndarray]]:
    atlas_flat = np.asarray(atlas_gray).ravel().astype(float, copy=False)
    if atlas_flat.size != cortex_mask.size:
        raise ValueError(
            f"Atlas labels length ({atlas_flat.size}) does not match CIFTI brain axis size ({cortex_mask.size})"
        )

    use = cortex_mask & np.isfinite(atlas_flat)
    if ignore_zero:
        use = use & (atlas_flat != 0)

    if not np.any(use):
        return ([], [])

    labels_f = np.unique(atlas_flat[use])
    if np.all(np.isclose(labels_f, np.round(labels_f))):
        labels = [int(x) for x in np.round(labels_f).astype(int).tolist()]
    else:
        labels = [int(x) for x in labels_f.astype(int).tolist()]

    labels = sorted(set(labels))
    indices_per_label: list[np.ndarray] = []
    for lab in labels:
        indices_per_label.append(np.flatnonzero(use & (atlas_flat == float(lab))))

    return (labels, indices_per_label)


def _labels_to_vertex_masks(
    *,
    label_vertices: np.ndarray,
    ignore_zero: bool,
) -> tuple[list[int], dict[int, np.ndarray]]:
    flat = np.asarray(label_vertices, dtype=float).ravel()
    finite = np.isfinite(flat)
    if ignore_zero:
        finite &= flat != 0
    if not np.any(finite):
        return ([], {})

    labels_f = np.unique(flat[finite])
    if np.all(np.isclose(labels_f, np.round(labels_f))):
        labels = [int(x) for x in np.round(labels_f).astype(int).tolist()]
    else:
        labels = [int(x) for x in labels_f.astype(int).tolist()]

    labels = sorted(set(labels))
    masks: dict[int, np.ndarray] = {}
    for lab in labels:
        masks[lab] = np.asarray(flat == float(lab), dtype=bool)
    return (labels, masks)


def _labels_to_vertex_indices(
    *,
    label_vertices: np.ndarray,
    ignore_zero: bool,
) -> tuple[list[int], dict[int, np.ndarray]]:
    flat = np.asarray(label_vertices, dtype=float).ravel()
    finite = np.isfinite(flat)
    if ignore_zero:
        finite &= flat != 0
    if not np.any(finite):
        return ([], {})

    labels_f = np.unique(flat[finite])
    if np.all(np.isclose(labels_f, np.round(labels_f))):
        labels = [int(x) for x in np.round(labels_f).astype(int).tolist()]
    else:
        labels = [int(x) for x in labels_f.astype(int).tolist()]

    labels = sorted(set(labels))
    indices: dict[int, np.ndarray] = {}
    for lab in labels:
        indices[lab] = np.flatnonzero(flat == float(lab)).astype(np.int64, copy=False)
    return (labels, indices)


def _plot_label_stats_from_vertices(
    ax: Axes,
    *,
    left_values: np.ndarray,
    right_values: np.ndarray,
    labels: list[int],
    left_indices: dict[int, np.ndarray],
    right_indices: dict[int, np.ndarray],
    rng: np.random.Generator,
    max_points_per_label: int,
    point_alpha: float,
    jitter: float,
    title: str | None,
    ylim: tuple[float | None, float | None] | None,
) -> None:
    ax.clear()
    if len(labels) == 0:
        ax.text(0.5, 0.5, "No atlas labels to plot", ha="center", va="center")
        ax.set_axis_off()
        return

    left_flat = np.asarray(left_values, dtype=float).ravel()
    right_flat = np.asarray(right_values, dtype=float).ravel()

    bxp_stats: list[dict[str, float]] = []
    means: list[float] = []
    positions = np.arange(1, len(labels) + 1, dtype=float)

    for lab in labels:
        li = left_indices.get(lab)
        ri = right_indices.get(lab)

        vals_list: list[np.ndarray] = []
        if li is not None and li.size:
            vals_list.append(left_flat[li])
        if ri is not None and ri.size:
            vals_list.append(right_flat[ri])
        vals = (
            np.concatenate(vals_list, axis=0)
            if vals_list
            else np.asarray([], dtype=float)
        )
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            q1 = med = q3 = whislo = whishi = float("nan")
            mean = float("nan")
        else:
            q1, med, q3 = (float(x) for x in np.percentile(vals, [25, 50, 75]))
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            inliers_lo = vals[vals >= lo]
            inliers_hi = vals[vals <= hi]
            whislo = (
                float(np.min(inliers_lo)) if inliers_lo.size else float(np.min(vals))
            )
            whishi = (
                float(np.max(inliers_hi)) if inliers_hi.size else float(np.max(vals))
            )
            mean = float(np.mean(vals))

        bxp_stats.append(
            {
                "q1": q1,
                "med": med,
                "q3": q3,
                "whislo": whislo,
                "whishi": whishi,
            }
        )
        means.append(mean)

    ax.bxp(
        bxp_stats,
        positions=positions,
        widths=0.6,
        showfliers=False,
    )

    if max_points_per_label > 0 and point_alpha > 0:
        for pos, lab in zip(positions, labels, strict=True):
            li = left_indices.get(lab)
            ri = right_indices.get(lab)
            vals_list = []
            if li is not None and li.size:
                vals_list.append(left_flat[li])
            if ri is not None and ri.size:
                vals_list.append(right_flat[ri])
            vals = (
                np.concatenate(vals_list, axis=0)
                if vals_list
                else np.asarray([], dtype=float)
            )
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if vals.size > max_points_per_label:
                pick = rng.choice(vals.size, size=max_points_per_label, replace=False)
                vals = vals[pick]
            x = pos + rng.uniform(-jitter, jitter, size=vals.size)
            ax.scatter(
                x,
                vals,
                s=6,
                alpha=float(point_alpha),
                color="black",
                linewidths=0,
            )

    ax.plot(positions, means, linestyle="none", marker="o", markersize=3, color="red")

    ax.set_xlim(0.5, float(len(labels)) + 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [str(label_id) for label_id in labels],
        rotation=90 if len(labels) > 15 else 0,
    )
    ax.set_xlabel("Atlas label")
    ax.set_ylabel("Values")
    if title:
        ax.set_title(title)
    if ylim is not None:
        y0, y1 = ylim
        if y0 is not None and y1 is not None:
            ax.set_ylim(float(y0), float(y1))


def _plot_label_stats(
    ax: Axes,
    *,
    frame_vec: np.ndarray,
    labels: list[int],
    indices_per_label: list[np.ndarray],
    rng: np.random.Generator,
    max_points_per_label: int,
    point_alpha: float,
    jitter: float,
    title: str | None,
    ylim: tuple[float | None, float | None] | None,
) -> None:
    ax.clear()
    if len(labels) == 0:
        ax.text(0.5, 0.5, "No atlas labels to plot", ha="center", va="center")
        ax.set_axis_off()
        return

    frame_flat = np.asarray(frame_vec).ravel().astype(float, copy=False)

    bxp_stats: list[dict[str, float]] = []
    means: list[float] = []
    positions = np.arange(1, len(labels) + 1, dtype=float)

    for idx in indices_per_label:
        vals = frame_flat[idx]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            q1 = med = q3 = whislo = whishi = float("nan")
            mean = float("nan")
        else:
            q1, med, q3 = (float(x) for x in np.percentile(vals, [25, 50, 75]))
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            inliers_lo = vals[vals >= lo]
            inliers_hi = vals[vals <= hi]
            whislo = (
                float(np.min(inliers_lo)) if inliers_lo.size else float(np.min(vals))
            )
            whishi = (
                float(np.max(inliers_hi)) if inliers_hi.size else float(np.max(vals))
            )
            mean = float(np.mean(vals))

        bxp_stats.append(
            {
                "q1": q1,
                "med": med,
                "q3": q3,
                "whislo": whislo,
                "whishi": whishi,
            }
        )
        means.append(mean)

    ax.bxp(
        bxp_stats,
        positions=positions,
        widths=0.6,
        showfliers=False,
    )

    if max_points_per_label > 0 and point_alpha > 0:
        for pos, idx in zip(positions, indices_per_label, strict=True):
            vals = frame_flat[idx]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if vals.size > max_points_per_label:
                pick = rng.choice(vals.size, size=max_points_per_label, replace=False)
                vals = vals[pick]
            x = pos + rng.uniform(-jitter, jitter, size=vals.size)
            ax.scatter(
                x,
                vals,
                s=6,
                alpha=float(point_alpha),
                color="black",
                linewidths=0,
            )

    ax.plot(positions, means, linestyle="none", marker="o", markersize=3, color="red")

    ax.set_xlim(0.5, float(len(labels)) + 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [str(label_id) for label_id in labels],
        rotation=90 if len(labels) > 15 else 0,
    )
    ax.set_xlabel("Atlas label")
    ax.set_ylabel("Values")
    if title:
        ax.set_title(title)
    if ylim is not None:
        y0, y1 = ylim
        if y0 is not None and y1 is not None:
            ax.set_ylim(float(y0), float(y1))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render CIFTI-2 dtseries/dscalar to surface plots and optionally encode a movie.",
    )

    p.add_argument("--input", required=True, type=Path, help="Input CIFTI file")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output movie path (.mp4). Defaults to <input stem>.mp4",
    )
    p.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory to write frames. Defaults to a temp directory.",
    )
    p.add_argument(
        "--keep-frames",
        action="store_true",
        help="Do not delete frames directory when done (only applies to temp dir).",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Only render frames; do not run ffmpeg.",
    )

    # Surface / plotting controls
    p.add_argument(
        "--surf-left",
        type=Path,
        default=Path("templates/fsLR_den-32k_hemi-L_inflated.surf.gii"),
        help="Left hemisphere fsLR32k inflated surface GIFTI.",
    )
    p.add_argument(
        "--surf-right",
        type=Path,
        default=Path("templates/fsLR_den-32k_hemi-R_inflated.surf.gii"),
        help="Right hemisphere fsLR32k inflated surface GIFTI.",
    )
    p.add_argument(
        "--surf-views",
        nargs="+",
        default=["lateral", "medial"],
        help=(
            "One or more nilearn surface views (e.g., lateral medial dorsal ventral). "
            "Each view renders both hemispheres as separate panels."
        ),
    )
    p.add_argument(
        "--surf-zoom",
        type=float,
        default=1.8,
        help=(
            "Zoom factor for mplot3d surface panels (default: 1.8). "
            "Larger fills more of each subplot but can crop if too large."
        ),
    )

    p.add_argument(
        "--surf-elev-offset",
        type=float,
        default=0.0,
        help=(
            "Camera elevation offset in degrees (applied after nilearn sets the base view). "
            "Positive tilts the camera to look more from above; negative from below."
        ),
    )
    p.add_argument(
        "--surf-azim-offset",
        type=float,
        default=0.0,
        help=(
            "Camera azimuth offset in degrees (applied after nilearn sets the base view). "
            "Use this to rotate slightly anterior/posterior in a lateral view."
        ),
    )

    # Optional per-hemisphere overrides (useful to un-occlude sulci differently on L/R).
    p.add_argument(
        "--surf-elev-offset-left",
        type=float,
        default=None,
        help=(
            "Per-hemisphere override for camera elevation offset (LEFT). "
            "If not provided, falls back to --surf-elev-offset."
        ),
    )
    p.add_argument(
        "--surf-elev-offset-right",
        type=float,
        default=None,
        help=(
            "Per-hemisphere override for camera elevation offset (RIGHT). "
            "If not provided, falls back to --surf-elev-offset."
        ),
    )
    p.add_argument(
        "--surf-azim-offset-left",
        type=float,
        default=None,
        help=(
            "Per-hemisphere override for camera azimuth offset (LEFT). "
            "If not provided, falls back to --surf-azim-offset."
        ),
    )
    p.add_argument(
        "--surf-azim-offset-right",
        type=float,
        default=None,
        help=(
            "Per-hemisphere override for camera azimuth offset (RIGHT). "
            "If not provided, falls back to --surf-azim-offset."
        ),
    )

    p.add_argument(
        "--surf-offsets-scope",
        choices=["all", "roi", "full"],
        default="all",
        help=(
            "Which surface panels should receive the camera view offsets. "
            "'all' applies offsets to every surface panel (default). "
            "'roi' applies offsets only to ROI-focused panels (when ROI masks are provided). "
            "'full' applies offsets only to full-surface panels."
        ),
    )

    p.add_argument(
        "--roi-mask-left",
        type=Path,
        default=None,
        help=(
            "Optional left hemisphere ROI mask GIFTI (.func.gii or .label.gii). "
            "Nonzero vertices are treated as inside-ROI. When provided, views are forced to lateral."
        ),
    )
    p.add_argument(
        "--roi-mask-right",
        type=Path,
        default=None,
        help=(
            "Optional right hemisphere ROI mask GIFTI (.func.gii or .label.gii). "
            "Nonzero vertices are treated as inside-ROI. When provided, views are forced to lateral."
        ),
    )
    p.add_argument(
        "--roi-focus-pad-mm",
        type=float,
        default=5.0,
        help="Padding (mm) around ROI bounding box when cropping the 3D axes.",
    )
    p.add_argument(
        "--roi-show-full",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When ROI masks are provided, also render a full-surface view alongside the ROI-focused view. "
            "Full-surface panels use --surf-views; ROI panels are always lateral and cropped."
        ),
    )
    p.add_argument(
        "--ncols",
        type=int,
        default=None,
        help=(
            "Number of columns when rendering multiple surface panels. "
            "Defaults to all panels in one row (i.e., n_panels)."
        ),
    )

    p.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap name.")
    p.add_argument("--vmin", type=float, default=None, help="Lower bound for colormap.")
    p.add_argument("--vmax", type=float, default=None, help="Upper bound for colormap.")
    p.add_argument(
        "--intensity-mode",
        choices=["global", "frame"],
        default="global",
        help=(
            "How to set vmin/vmax when not explicitly provided. "
            "'global' (default) uses robust percentiles across all selected frames; "
            "'frame' recomputes percentiles per frame."
        ),
    )
    p.add_argument(
        "--auto-percentiles",
        nargs=2,
        type=float,
        metavar=("PLOW", "PHIGH"),
        default=(1.0, 99.0),
        help="Percentiles for automatic vmin/vmax (default: 1 99).",
    )
    p.add_argument(
        "--auto-max-samples",
        type=int,
        default=200_000,
        help="Max random samples per frame when estimating percentiles (default: 200000).",
    )
    p.add_argument(
        "--auto-max-total-samples",
        type=int,
        default=2_000_000,
        help=(
            "Max total random samples used to estimate global percentiles (default: 2000000). "
            "Larger is more accurate but slower."
        ),
    )
    p.add_argument(
        "--black-bg",
        action="store_true",
        help="Use black figure background.",
    )
    p.add_argument(
        "--colorbar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show colorbar (default: true).",
    )
    p.add_argument(
        "--colorbar-side",
        choices=["right", "left"],
        default="right",
        help="Where to place the colorbar when enabled (default: right).",
    )
    p.add_argument(
        "--title",
        default=None,
        help=(
            "Optional title template. Use {frame}, {index}, {panel}, {view}, {hemi}, {time}, {focus} placeholders. "
            "(focus is 'full' or 'roi'.)"
        ),
    )

    # Optional atlas stats
    p.add_argument(
        "--atlas",
        type=Path,
        default=None,
        help=(
            "Optional CIFTI dense label atlas (.dlabel.nii). If provided, a dynamic "
            "per-label stats subplot is added below the surface maps."
        ),
    )
    p.add_argument(
        "--atlas-ignore-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat atlas label 0 as background (default: true).",
    )
    p.add_argument(
        "--atlas-stats-max-points-per-label",
        type=int,
        default=300,
        help="Max scatter points per atlas label (default: 300). Set 0 to disable scatter.",
    )
    p.add_argument(
        "--atlas-stats-point-alpha",
        type=float,
        default=0.08,
        help="Alpha for scatter points in atlas stats subplot (default: 0.08).",
    )
    p.add_argument(
        "--atlas-stats-jitter",
        type=float,
        default=0.18,
        help="Horizontal jitter for scatter points (default: 0.18).",
    )

    # Time annotation
    p.add_argument(
        "--time-annotate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Annotate each frame with time in seconds (default: false).",
    )
    p.add_argument(
        "--tr",
        type=float,
        default=None,
        help="Repetition time in seconds (required if --time-annotate).",
    )
    p.add_argument(
        "--t0-trs",
        type=float,
        default=0.0,
        help="Starting time in TRs for index 0 (default: 0.0; can be negative).",
    )

    # Frame / resolution controls
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved frames (default: 150).",
    )
    p.add_argument(
        "--size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(1280, 720),
        help="Frame size in pixels before encoding (default: 1280 720).",
    )

    # Frame indices
    p.add_argument(
        "--start", type=int, default=0, help="First frame index (default: 0)."
    )
    p.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Stop frame index (exclusive). Defaults to end.",
    )
    p.add_argument("--step", type=int, default=1, help="Frame step (default: 1).")

    # Video encoding
    p.add_argument(
        "--fps", type=float, default=10.0, help="Frames per second (default: 10)."
    )
    p.add_argument(
        "--ffmpeg",
        type=str,
        default=None,
        help="Path to ffmpeg executable (optional).",
    )
    p.add_argument(
        "--crf",
        type=int,
        default=23,
        help="x264 CRF quality (lower=better, larger files). Default: 23.",
    )
    p.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        help="x264 preset (default: medium).",
    )
    p.add_argument(
        "--scale",
        type=str,
        default=None,
        help="Optional ffmpeg scale W:H (e.g. 960:-2) to downscale during encoding.",
    )

    p.add_argument(
        "--fast-render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "(Experimental) Reuse surface artists across frames. "
            "May or may not be faster depending on Matplotlib/Nilearn versions. "
            "Automatically falls back to slow rendering when per-frame intensity scaling is used."
        ),
    )

    p.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print periodic progress updates (default: true).",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if bool(args.time_annotate) and args.tr is None:
        raise ValueError("--tr is required when using --time-annotate")

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    if not Path(args.surf_left).exists():
        raise FileNotFoundError(str(args.surf_left))
    if not Path(args.surf_right).exists():
        raise FileNotFoundError(str(args.surf_right))

    # Pre-load surface meshes once. Passing an in-memory mesh is much faster than
    # letting nilearn re-load and parse the GIFTI file for every panel.
    surf_left_mesh = surface.load_surf_mesh(str(args.surf_left))
    surf_right_mesh = surface.load_surf_mesh(str(args.surf_right))

    # Optional ROI focus masks (full vertex space). If provided, enforce lateral view.
    roi_mask_L: np.ndarray | None = None
    roi_mask_R: np.ndarray | None = None
    roi_focus = bool(args.roi_mask_left) or bool(args.roi_mask_right)
    if args.roi_mask_left is not None:
        if not Path(args.roi_mask_left).exists():
            raise FileNotFoundError(str(args.roi_mask_left))
        roi_mask_L = _load_vertex_mask_gifti(Path(args.roi_mask_left))
    if args.roi_mask_right is not None:
        if not Path(args.roi_mask_right).exists():
            raise FileNotFoundError(str(args.roi_mask_right))
        roi_mask_R = _load_vertex_mask_gifti(Path(args.roi_mask_right))

    # Validate mask lengths against the loaded surfaces.
    if roi_mask_L is not None:
        left_coords, _left_faces = surf_left_mesh
        if int(roi_mask_L.size) != int(np.asarray(left_coords).shape[0]):
            raise ValueError(
                "Left ROI mask length does not match left surface vertices: "
                f"mask={int(roi_mask_L.size)} vs surf={int(np.asarray(left_coords).shape[0])}"
            )
    if roi_mask_R is not None:
        right_coords, _right_faces = surf_right_mesh
        if int(roi_mask_R.size) != int(np.asarray(right_coords).shape[0]):
            raise ValueError(
                "Right ROI mask length does not match right surface vertices: "
                f"mask={int(roi_mask_R.size)} vs surf={int(np.asarray(right_coords).shape[0])}"
            )

    if roi_focus and (not bool(args.roi_show_full)):
        requested = [str(v) for v in (args.surf_views or [])]
        if requested != ["lateral"]:
            print(
                f"[cifti_movie] ROI focus enabled; overriding --surf-views {requested!r} -> ['lateral']",
                file=sys.stderr,
            )
        args.surf_views = ["lateral"]

    output_path: Path
    if args.output is None:
        stem = input_path.name
        # Preserve common CIFTI suffixes for nicer output names
        for suffix in (
            ".dtseries.nii",
            ".dscalar.nii",
            ".dconn.nii",
            ".nii.gz",
            ".nii",
        ):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        output_path = input_path.with_name(f"{stem}.mp4")
    else:
        output_path = args.output

    img = nib_load(str(input_path))
    if not isinstance(img, nib.cifti2.cifti2.Cifti2Image):
        raise ValueError(f"Expected a CIFTI-2 image, got {type(img)}")

    frame_axis, brain_axis, frame_spec = _infer_cifti_axes(img)
    n_frames = int(frame_spec.n_frames)
    frame_indices = list(
        _iter_frame_indices(n_frames, args.start, args.stop, args.step)
    )
    if len(frame_indices) == 0:
        raise ValueError("No frames selected")

    structures = _extract_cortex_structures(brain_axis)
    cortex_mask = _compute_cortex_mask(brain_axis, structures)
    if not np.any(cortex_mask):
        raise ValueError("No cortex structures found in CIFTI brain models")

    # Optional atlas labels in grayordinate order
    atlas_labels: list[int] = []
    atlas_left_indices: dict[int, np.ndarray] = {}
    atlas_right_indices: dict[int, np.ndarray] = {}
    if args.atlas is not None:
        if not Path(args.atlas).exists():
            raise FileNotFoundError(str(args.atlas))
        atlas_img = nib_load(str(args.atlas))
        if not isinstance(atlas_img, nib.cifti2.cifti2.Cifti2Image):
            raise ValueError(f"Expected a CIFTI-2 atlas image, got {type(atlas_img)}")

        a0 = atlas_img.header.get_index_map(0)
        a1 = atlas_img.header.get_index_map(1)
        from nibabel.cifti2 import cifti2_axes

        ax0 = cifti2_axes.from_index_mapping(a0)
        ax1 = cifti2_axes.from_index_mapping(a1)
        if isinstance(ax0, cifti2_axes.BrainModelAxis) and not isinstance(
            ax1, cifti2_axes.BrainModelAxis
        ):
            atlas_brain = ax0
            atlas_vec = (
                np.asanyarray(atlas_img.dataobj[:, 0]).ravel().astype(float, copy=False)
            )
        elif isinstance(ax1, cifti2_axes.BrainModelAxis) and not isinstance(
            ax0, cifti2_axes.BrainModelAxis
        ):
            atlas_brain = ax1
            atlas_vec = (
                np.asanyarray(atlas_img.dataobj[0, :]).ravel().astype(float, copy=False)
            )
        else:
            raise ValueError(
                "Expected atlas to have one BrainModelAxis and one label axis; "
                f"got axis0={type(ax0)} axis1={type(ax1)}"
            )

        # We only use cortex for both plotting and stats, so the atlas can be:
        # - cortex-only (common for fsLR32k dense labels), OR
        # - full grayordinates (cortex+subcortex). In both cases, we extract
        #   left/right cortex label arrays and build per-label vertex masks.
        atlas_structures = _extract_cortex_structures(atlas_brain)

        atlas_left = _brain_to_hemi_vertices(
            frame_vec=atlas_vec,
            structures=atlas_structures,
            structure_name="CIFTI_STRUCTURE_CORTEX_LEFT",
        )
        atlas_right = _brain_to_hemi_vertices(
            frame_vec=atlas_vec,
            structures=atlas_structures,
            structure_name="CIFTI_STRUCTURE_CORTEX_RIGHT",
        )

        labels_l, idx_l = _labels_to_vertex_indices(
            label_vertices=atlas_left,
            ignore_zero=bool(args.atlas_ignore_zero),
        )
        labels_r, idx_r = _labels_to_vertex_indices(
            label_vertices=atlas_right,
            ignore_zero=bool(args.atlas_ignore_zero),
        )

        atlas_labels = sorted(set(labels_l).union(labels_r))
        atlas_left_indices = idx_l
        atlas_right_indices = idx_r

    # Determine vmin/vmax behavior.
    p_low, p_high = _validate_percentiles(
        float(args.auto_percentiles[0]), float(args.auto_percentiles[1])
    )

    global_vmin: float | None = None
    global_vmax: float | None = None
    if args.vmin is None or args.vmax is None:
        if args.intensity_mode == "global":
            rng = np.random.default_rng(0)
            max_total = int(args.auto_max_total_samples)
            samples_per_frame = max(1, int(max_total // max(len(frame_indices), 1)))
            collected: list[np.ndarray] = []
            for fi in frame_indices:
                frame_vec = _get_frame_vector(
                    img, frame_index=int(fi), frame_spec=frame_spec
                )
                frame_vec = frame_vec[cortex_mask]
                collected.append(
                    _sample_finite_values(
                        frame_vec, max_samples=samples_per_frame, rng=rng
                    )
                )
            all_samples = (
                np.concatenate([s for s in collected if s.size > 0], axis=0)
                if collected
                else np.asarray([], dtype=float)
            )
            if all_samples.size > 0:
                global_vmin = float(np.percentile(all_samples, p_low))
                global_vmax = float(np.percentile(all_samples, p_high))

    width_px, height_px = int(args.size[0]), int(args.size[1])
    dpi = int(args.dpi)
    figsize = (width_px / dpi, height_px / dpi)

    # Prepare output frames directory
    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.frames_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="cifti_frames_")
        frames_dir = Path(temp_dir_obj.name)
    else:
        frames_dir = args.frames_dir
        frames_dir.mkdir(parents=True, exist_ok=True)

    keep_temp_frames = bool(args.keep_frames) or bool(args.no_video)

    views_full = [str(v) for v in (args.surf_views or [])]
    views_roi = ["lateral"] if roi_focus else []

    # Panels = (view, hemi, focus) where focus is 'full' or 'roi'.
    panels: list[tuple[str, str, str]] = []

    for view in views_full:
        panels.append((view, "L", "full"))
        panels.append((view, "R", "full"))

    if roi_focus:
        if not bool(args.roi_show_full):
            # ROI-only layout
            panels = []
        for view in views_roi:
            panels.append((view, "L", "roi"))
            panels.append((view, "R", "roi"))

    can_fast_render = bool(args.fast_render)
    # Fast path assumes vmin/vmax are stable across frames because it keeps a
    # single colorbar and avoids re-initializing axes. If intensity-mode is
    # "frame" and vmin/vmax aren't user-specified, we must fall back.
    if args.intensity_mode == "frame" and (args.vmin is None or args.vmax is None):
        can_fast_render = False

    try:
        frame_rng = np.random.default_rng(0)
        stats_rng = np.random.default_rng(0)
        t_all0 = time.perf_counter()

        n_panels = len(panels)
        if n_panels <= 0:
            raise RuntimeError("No surface panels to render")

        ncols = int(args.ncols) if args.ncols is not None else int(n_panels)
        if ncols <= 0:
            raise ValueError(f"--ncols must be positive, got {ncols}")
        nrows_maps = int(np.ceil(n_panels / ncols))

        show_stats = args.atlas is not None
        want_cbar = bool(args.colorbar)
        cbar_side = str(args.colorbar_side)
        ncols_total = int(ncols + (1 if want_cbar else 0))
        surf_col0 = 1 if (want_cbar and cbar_side == "left") else 0
        cbar_col = 0 if (want_cbar and cbar_side == "left") else (ncols_total - 1)
        cbar_width_ratio = 0.08
        width_ratios = None
        if want_cbar:
            if cbar_side == "left":
                width_ratios = [cbar_width_ratio] + [1.0] * ncols
            else:
                width_ratios = [1.0] * ncols + [cbar_width_ratio]

        if can_fast_render:
            # Reuse a single figure/axes across all frames and only update
            # Poly3DCollection facecolors.
            fig = plt.figure(figsize=figsize)
            if bool(args.black_bg):
                fig.patch.set_facecolor("black")

            if show_stats:
                height_ratios = [1.0] * nrows_maps + [0.6]
                gs = fig.add_gridspec(
                    nrows=nrows_maps + 1,
                    ncols=ncols_total,
                    height_ratios=height_ratios,
                    width_ratios=width_ratios,
                    wspace=0.02,
                    hspace=0.0,
                )
            else:
                gs = fig.add_gridspec(
                    nrows=nrows_maps,
                    ncols=ncols_total,
                    width_ratios=width_ratios,
                    wspace=0.02,
                    hspace=0.0,
                )

            axes_flat: list[Axes] = []
            for r in range(nrows_maps):
                for c in range(ncols):
                    axes_flat.append(
                        fig.add_subplot(gs[r, c + surf_col0], projection="3d")
                    )

            ax_stats = (
                fig.add_subplot(gs[-1, surf_col0 : surf_col0 + ncols])
                if show_stats
                else None
            )
            ax_cbar = None
            if want_cbar:
                ax_cbar_container = fig.add_subplot(gs[:nrows_maps, cbar_col])
                ax_cbar_container.set_axis_off()
                ax_cbar = inset_axes(
                    ax_cbar_container,
                    width="55%",
                    height="70%",
                    loc="center",
                    borderpad=0.0,
                )

            # Precompute faces + background facecolors for each hemi.
            left_coords, left_faces = surf_left_mesh
            right_coords, right_faces = surf_right_mesh
            left_faces = np.asarray(left_faces, dtype=np.int64)
            right_faces = np.asarray(right_faces, dtype=np.int64)
            bg_alpha = 0.5
            bg_left = _compute_bg_facecolors(
                n_faces=int(left_faces.shape[0]), alpha=bg_alpha
            )
            bg_right = _compute_bg_facecolors(
                n_faces=int(right_faces.shape[0]), alpha=bg_alpha
            )

            cmap_obj = plt.get_cmap(str(args.cmap))

            time_text = None
            if bool(args.time_annotate):
                color = "white" if bool(args.black_bg) else "black"
                time_text = fig.text(0.01, 0.99, "", ha="left", va="top", color=color)

            # Initialize surface panels using the first frame.
            init_frame_index = int(frame_indices[0])
            init_vec = _get_frame_vector(
                img, frame_index=init_frame_index, frame_spec=frame_spec
            )
            init_left = _brain_to_hemi_vertices(
                frame_vec=init_vec,
                structures=structures,
                structure_name="CIFTI_STRUCTURE_CORTEX_LEFT",
            )
            init_right = _brain_to_hemi_vertices(
                frame_vec=init_vec,
                structures=structures,
                structure_name="CIFTI_STRUCTURE_CORTEX_RIGHT",
            )

            # Determine fixed vmin/vmax for fast mode.
            if args.vmin is not None and args.vmax is not None:
                fixed_vmin = float(args.vmin)
                fixed_vmax = float(args.vmax)
            else:
                fixed_vmin = (
                    float(global_vmin)
                    if global_vmin is not None
                    else float(np.nanmin(init_vec[cortex_mask]))
                )
                fixed_vmax = (
                    float(global_vmax)
                    if global_vmax is not None
                    else float(np.nanmax(init_vec[cortex_mask]))
                )

            panel_polys: list[Any] = []
            panel_faces: list[np.ndarray] = []
            panel_bg: list[np.ndarray] = []
            panel_hemi_kw: list[str] = []

            for panel_idx, (view, hemi, focus) in enumerate(panels):
                ax = axes_flat[panel_idx]
                if hemi == "L":
                    stat_map = init_left
                    surf_mesh = surf_left_mesh
                    hemi_kw = "left"
                    faces = left_faces
                    bg = bg_left
                else:
                    stat_map = init_right
                    surf_mesh = surf_right_mesh
                    hemi_kw = "right"
                    faces = right_faces
                    bg = bg_right

                title = None
                if args.title:
                    time_sec = (
                        (float(args.t0_trs) + float(init_frame_index)) * float(args.tr)
                        if args.tr is not None
                        else None
                    )
                    title = str(args.title).format(
                        frame=0,
                        index=init_frame_index,
                        panel=panel_idx,
                        view=view,
                        hemi=hemi,
                        focus=focus,
                        time=time_sec,
                    )

                plotting.plot_surf_stat_map(
                    surf_mesh,
                    stat_map,
                    hemi=hemi_kw,
                    view=str(view),
                    cmap=str(args.cmap),
                    vmin=fixed_vmin,
                    vmax=fixed_vmax,
                    colorbar=False,
                    title=title,
                    figure=fig,
                    axes=ax,
                )

                scope = str(args.surf_offsets_scope)
                apply_offsets = True
                if scope == "roi" and focus != "roi":
                    apply_offsets = False
                elif scope == "full" and focus != "full":
                    apply_offsets = False

                if apply_offsets:
                    elev_off = (
                        float(args.surf_elev_offset_left)
                        if (hemi == "L" and args.surf_elev_offset_left is not None)
                        else float(args.surf_elev_offset_right)
                        if (hemi == "R" and args.surf_elev_offset_right is not None)
                        else float(args.surf_elev_offset)
                    )
                    azim_off = (
                        float(args.surf_azim_offset_left)
                        if (hemi == "L" and args.surf_azim_offset_left is not None)
                        else float(args.surf_azim_offset_right)
                        if (hemi == "R" and args.surf_azim_offset_right is not None)
                        else float(args.surf_azim_offset)
                    )
                    _apply_surf_view_offsets(
                        ax,
                        elev_offset=elev_off,
                        azim_offset=azim_off,
                    )

                # Edges are expensive to rasterize in mplot3d; disabling them
                # can noticeably speed up frame rendering.
                if ax.collections:
                    try:
                        ax.collections[0].set_edgecolor("none")
                        ax.collections[0].set_linewidth(0)
                    except Exception:
                        pass

                _apply_surf_zoom(ax, float(args.surf_zoom))

                # If ROI focus is enabled, crop axes to ROI bounding box.
                if focus == "roi" and str(view) == "lateral":
                    if hemi == "L" and roi_mask_L is not None:
                        coords_L, _ = surf_left_mesh
                        _focus_3d_axes_on_mask(
                            ax,
                            coords=np.asarray(coords_L),
                            mask=roi_mask_L,
                            pad_mm=float(args.roi_focus_pad_mm),
                        )
                    elif hemi == "R" and roi_mask_R is not None:
                        coords_R, _ = surf_right_mesh
                        _focus_3d_axes_on_mask(
                            ax,
                            coords=np.asarray(coords_R),
                            mask=roi_mask_R,
                            pad_mm=float(args.roi_focus_pad_mm),
                        )
                fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.06)

                poly = ax.collections[0] if ax.collections else None
                if poly is None:
                    raise RuntimeError("Failed to initialize surface collection")
                panel_polys.append(poly)
                panel_faces.append(faces)
                panel_bg.append(bg)
                panel_hemi_kw.append(hemi_kw)

            for ax in axes_flat[n_panels:]:
                ax.axis("off")

            if ax_cbar is not None:
                _add_colorbar(
                    fig=fig,
                    cax=ax_cbar,
                    cmap=str(args.cmap),
                    vmin=float(fixed_vmin),
                    vmax=float(fixed_vmax),
                )

            # Deterministic layout; important for 3D axes.
            fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.06)

            # Render all frames by updating existing artists.
            for frame_number, frame_index in enumerate(frame_indices):
                t0 = time.perf_counter()
                fi = int(frame_index)
                frame_vec = _get_frame_vector(
                    img, frame_index=fi, frame_spec=frame_spec
                )
                left_vertices = _brain_to_hemi_vertices(
                    frame_vec=frame_vec,
                    structures=structures,
                    structure_name="CIFTI_STRUCTURE_CORTEX_LEFT",
                )
                right_vertices = _brain_to_hemi_vertices(
                    frame_vec=frame_vec,
                    structures=structures,
                    structure_name="CIFTI_STRUCTURE_CORTEX_RIGHT",
                )

                if args.tr is None:
                    time_sec = None
                else:
                    time_sec = (float(args.t0_trs) + float(fi)) * float(args.tr)

                if time_text is not None:
                    label = "t = ? s" if time_sec is None else f"t = {time_sec:.3f} s"
                    time_text.set_text(label)

                # Update surface colors.
                for panel_idx, (view, hemi, focus) in enumerate(panels):
                    poly = panel_polys[panel_idx]
                    faces = panel_faces[panel_idx]
                    bg = panel_bg[panel_idx]
                    stat_map = left_vertices if hemi == "L" else right_vertices

                    title = None
                    if args.title:
                        title = str(args.title).format(
                            frame=frame_number,
                            index=fi,
                            panel=panel_idx,
                            view=view,
                            hemi=hemi,
                            focus=focus,
                            time=time_sec,
                        )
                        axes_flat[panel_idx].set_title(title)

                    _update_poly3d_facecolors(
                        poly=poly,
                        faces=faces,
                        stat_map_vertices=stat_map,
                        cmap=cmap_obj,
                        vmin=fixed_vmin,
                        vmax=fixed_vmax,
                        bg_facecolors=bg,
                    )

                if ax_stats is not None:
                    stats_title = (
                        f"Atlas distributions (t={time_sec:.3f}s)"
                        if time_sec is not None
                        else f"Atlas distributions (index={fi})"
                    )
                    _plot_label_stats_from_vertices(
                        ax_stats,
                        left_values=left_vertices,
                        right_values=right_vertices,
                        labels=atlas_labels,
                        left_indices=atlas_left_indices,
                        right_indices=atlas_right_indices,
                        rng=stats_rng,
                        max_points_per_label=int(args.atlas_stats_max_points_per_label),
                        point_alpha=float(args.atlas_stats_point_alpha),
                        jitter=float(args.atlas_stats_jitter),
                        title=stats_title,
                        ylim=(fixed_vmin, fixed_vmax),
                    )

                frame_path = frames_dir / f"frame_{frame_number:05d}.png"
                fig.savefig(str(frame_path), dpi=dpi, facecolor=fig.get_facecolor())

                if bool(args.progress):
                    n_done = frame_number + 1
                    n_total = len(frame_indices)
                    if n_done == 1 or (n_done % 5) == 0 or n_done == n_total:
                        elapsed = time.perf_counter() - t0
                        total_elapsed = time.perf_counter() - t_all0
                        avg = total_elapsed / max(n_done, 1)
                        remaining = avg * (n_total - n_done)
                        print(
                            f"Rendered frame {n_done}/{n_total} (index={fi}) in {elapsed:.2f}s; "
                            f"avg {avg:.2f}s/frame; ETA {remaining / 60:.1f} min"
                        )

            plt.close(fig)

        else:
            # Slow-but-robust rendering: build a fresh figure per frame.
            for frame_number, frame_index in enumerate(frame_indices):
                t0 = time.perf_counter()
                frame_vec = _get_frame_vector(
                    img, frame_index=int(frame_index), frame_spec=frame_spec
                )

                # Cortex-only vertex arrays (ignore subcortex for visualization/stats).
                left_vertices = _brain_to_hemi_vertices(
                    frame_vec=frame_vec,
                    structures=structures,
                    structure_name="CIFTI_STRUCTURE_CORTEX_LEFT",
                )
                right_vertices = _brain_to_hemi_vertices(
                    frame_vec=frame_vec,
                    structures=structures,
                    structure_name="CIFTI_STRUCTURE_CORTEX_RIGHT",
                )

                time_sec: float | None
                if args.tr is None:
                    time_sec = None
                else:
                    time_sec = (float(args.t0_trs) + float(frame_index)) * float(
                        args.tr
                    )

                if args.vmin is not None and args.vmax is not None:
                    frame_vmin = float(args.vmin)
                    frame_vmax = float(args.vmax)
                else:
                    if args.intensity_mode == "frame":
                        samples = _sample_finite_values(
                            frame_vec[cortex_mask],
                            max_samples=int(args.auto_max_samples),
                            rng=frame_rng,
                        )
                        if samples.size > 0:
                            auto_vmin = float(np.percentile(samples, p_low))
                            auto_vmax = float(np.percentile(samples, p_high))
                        else:
                            auto_vmin = None
                            auto_vmax = None
                    else:
                        auto_vmin = global_vmin
                        auto_vmax = global_vmax

                    frame_vmin = (
                        float(args.vmin) if args.vmin is not None else auto_vmin
                    )
                    frame_vmax = (
                        float(args.vmax) if args.vmax is not None else auto_vmax
                    )

                fig = plt.figure(figsize=figsize)
                if bool(args.black_bg):
                    fig.patch.set_facecolor("black")

                if show_stats:
                    height_ratios = [1.0] * nrows_maps + [0.6]
                    gs = fig.add_gridspec(
                        nrows=nrows_maps + 1,
                        ncols=ncols_total,
                        height_ratios=height_ratios,
                        width_ratios=width_ratios,
                        wspace=0.02,
                        hspace=0.0,
                    )
                else:
                    gs = fig.add_gridspec(
                        nrows=nrows_maps,
                        ncols=ncols_total,
                        width_ratios=width_ratios,
                        wspace=0.02,
                        hspace=0.0,
                    )

                axes_flat: list[Axes] = []
                for r in range(nrows_maps):
                    for c in range(ncols):
                        axes_flat.append(
                            fig.add_subplot(gs[r, c + surf_col0], projection="3d")
                        )

                ax_stats = (
                    fig.add_subplot(gs[-1, surf_col0 : surf_col0 + ncols])
                    if show_stats
                    else None
                )
                ax_cbar = None
                if want_cbar:
                    ax_cbar_container = fig.add_subplot(gs[:nrows_maps, cbar_col])
                    ax_cbar_container.set_axis_off()
                    ax_cbar = inset_axes(
                        ax_cbar_container,
                        width="55%",
                        height="70%",
                        loc="center",
                        borderpad=0.0,
                    )

                for panel_idx, (view, hemi, focus) in enumerate(panels):
                    ax = axes_flat[panel_idx]

                    if hemi == "L":
                        stat_map = left_vertices
                        surf_mesh = surf_left_mesh
                        hemi_kw = "left"
                    else:
                        stat_map = right_vertices
                        surf_mesh = surf_right_mesh
                        hemi_kw = "right"

                    title = None
                    if args.title:
                        title = str(args.title).format(
                            frame=frame_number,
                            index=frame_index,
                            panel=panel_idx,
                            view=view,
                            hemi=hemi,
                            focus=focus,
                            time=time_sec,
                        )

                    plotting.plot_surf_stat_map(
                        surf_mesh,
                        stat_map,
                        hemi=hemi_kw,
                        view=str(view),
                        cmap=str(args.cmap),
                        vmin=frame_vmin,
                        vmax=frame_vmax,
                        colorbar=False,
                        title=title,
                        figure=fig,
                        axes=ax,
                    )

                    scope = str(args.surf_offsets_scope)
                    apply_offsets = True
                    if scope == "roi" and focus != "roi":
                        apply_offsets = False
                    elif scope == "full" and focus != "full":
                        apply_offsets = False

                    if apply_offsets:
                        elev_off = (
                            float(args.surf_elev_offset_left)
                            if (hemi == "L" and args.surf_elev_offset_left is not None)
                            else float(args.surf_elev_offset_right)
                            if (hemi == "R" and args.surf_elev_offset_right is not None)
                            else float(args.surf_elev_offset)
                        )
                        azim_off = (
                            float(args.surf_azim_offset_left)
                            if (hemi == "L" and args.surf_azim_offset_left is not None)
                            else float(args.surf_azim_offset_right)
                            if (hemi == "R" and args.surf_azim_offset_right is not None)
                            else float(args.surf_azim_offset)
                        )
                        _apply_surf_view_offsets(
                            ax,
                            elev_offset=elev_off,
                            azim_offset=azim_off,
                        )

                    if ax.collections:
                        try:
                            ax.collections[0].set_edgecolor("none")
                            ax.collections[0].set_linewidth(0)
                        except Exception:
                            pass

                    _apply_surf_zoom(ax, float(args.surf_zoom))

                    # If ROI focus is enabled, crop axes to ROI bounding box.
                    if focus == "roi" and str(view) == "lateral":
                        if hemi == "L" and roi_mask_L is not None:
                            coords_L, _ = surf_left_mesh
                            _focus_3d_axes_on_mask(
                                ax,
                                coords=np.asarray(coords_L),
                                mask=roi_mask_L,
                                pad_mm=float(args.roi_focus_pad_mm),
                            )
                        elif hemi == "R" and roi_mask_R is not None:
                            coords_R, _ = surf_right_mesh
                            _focus_3d_axes_on_mask(
                                ax,
                                coords=np.asarray(coords_R),
                                mask=roi_mask_R,
                                pad_mm=float(args.roi_focus_pad_mm),
                            )

                for ax in axes_flat[n_panels:]:
                    ax.axis("off")

                if ax_cbar is not None:
                    # Ensure we have finite bounds; nilearn will auto-scale if
                    # vmin/vmax were None, but we need explicit values here.
                    if frame_vmin is None or frame_vmax is None:
                        finite = frame_vec[cortex_mask]
                        finite = finite[np.isfinite(finite)]
                        if finite.size:
                            frame_vmin = float(np.percentile(finite, p_low))
                            frame_vmax = float(np.percentile(finite, p_high))
                        else:
                            frame_vmin, frame_vmax = (-1.0, 1.0)

                    _add_colorbar(
                        fig=fig,
                        cax=ax_cbar,
                        cmap=str(args.cmap),
                        vmin=float(frame_vmin),
                        vmax=float(frame_vmax),
                    )

                if ax_stats is not None:
                    stats_title = None
                    if time_sec is not None:
                        stats_title = f"Atlas distributions (t={time_sec:.3f}s)"
                    else:
                        stats_title = f"Atlas distributions (index={frame_index})"

                    _plot_label_stats_from_vertices(
                        ax_stats,
                        left_values=left_vertices,
                        right_values=right_vertices,
                        labels=atlas_labels,
                        left_indices=atlas_left_indices,
                        right_indices=atlas_right_indices,
                        rng=stats_rng,
                        max_points_per_label=int(args.atlas_stats_max_points_per_label),
                        point_alpha=float(args.atlas_stats_point_alpha),
                        jitter=float(args.atlas_stats_jitter),
                        title=stats_title,
                        ylim=(frame_vmin, frame_vmax)
                        if (frame_vmin is not None and frame_vmax is not None)
                        else None,
                    )

                fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.06)

                if bool(args.time_annotate):
                    label = "t = ? s" if time_sec is None else f"t = {time_sec:.3f} s"
                    color = "white" if bool(args.black_bg) else "black"
                    fig.text(0.01, 0.99, label, ha="left", va="top", color=color)

                frame_path = frames_dir / f"frame_{frame_number:05d}.png"
                fig.savefig(str(frame_path), dpi=dpi, facecolor=fig.get_facecolor())
                plt.close(fig)

                if bool(args.progress):
                    n_done = frame_number + 1
                    n_total = len(frame_indices)
                    if n_done == 1 or (n_done % 5) == 0 or n_done == n_total:
                        elapsed = time.perf_counter() - t0
                        total_elapsed = time.perf_counter() - t_all0
                        avg = total_elapsed / max(n_done, 1)
                        remaining = avg * (n_total - n_done)
                        print(
                            f"Rendered frame {n_done}/{n_total} (index={int(frame_index)}) in {elapsed:.2f}s; "
                            f"avg {avg:.2f}s/frame; ETA {remaining / 60:.1f} min"
                        )

        if args.no_video:
            print(f"Rendered {len(frame_indices)} frame(s) to: {frames_dir}")
            return 0

        ffmpeg = _find_ffmpeg(args.ffmpeg)
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg not found. Install system ffmpeg, or add optional extra 'viz' "
                "(imageio-ffmpeg) and re-run. You can also use --no-video to only render frames."
            )

        frames_pattern = str(frames_dir / "frame_%05d.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        _run_ffmpeg(
            ffmpeg=ffmpeg,
            frames_pattern=frames_pattern,
            fps=float(args.fps),
            output=output_path,
            crf=int(args.crf),
            preset=str(args.preset),
            scale=args.scale,
        )

        print(f"Wrote movie: {output_path}")
        return 0
    finally:
        if temp_dir_obj is not None and (not keep_temp_frames):
            temp_dir_obj.cleanup()
        elif temp_dir_obj is not None and keep_temp_frames:
            print(f"Kept frames at: {frames_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
