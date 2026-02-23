"""Render a NIfTI image (3D/4D) to frames and optionally encode an MP4.

This script is intended for quick visualization and QC. It uses nilearn to
render an orthogonal (sag/cor/ax) view by default.

Video encoding uses an ffmpeg executable. The script will try, in order:
1) --ffmpeg path (if provided)
2) system ffmpeg on PATH
3) imageio-ffmpeg (if installed; see optional dependency extra `viz`)

Examples
--------
# 4D movie (time axis), automatic cut coords
python scripts/nifti_movie.py --input sub-01_task-rest_bold.nii.gz --output bold.mp4 --fps 10


# Add an atlas summary subplot (box+scatter per atlas label) below the voxel maps
python scripts/nifti_movie.py --input bold.nii.gz --output bold_stats.mp4 \
    --atlas atlas_parcellation.nii.gz --atlas-stats-max-points-per-label 300

# Two orthogonal views (two cut locations) in the same frame
python scripts/nifti_movie.py --input bold.nii.gz --output bold_2views.mp4 \
    --cut-coords-list 0 -25 50  0 -10 40 --ncols 2

# Custom voxel slice indices (i j k), fixed intensity range, smaller output
python scripts/nifti_movie.py --input stat.nii.gz --output stat.mp4 \
  --slice-indices 40 52 36 --vmin -3 --vmax 3 --cmap cold_hot --size 960 720 --crf 28

# Render frames only (no ffmpeg needed)
python scripts/nifti_movie.py --input img.nii.gz --frames-dir frames --keep-frames --no-video

 
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from nibabel.loadsave import load as nib_load
import numpy as np
from nilearn import image, plotting


@dataclasses.dataclass(frozen=True)
class CutCoords:
    x: float | None
    y: float | None
    z: float | None

    def as_tuple_or_none(self) -> tuple[float, float, float] | None:
        if self.x is None or self.y is None or self.z is None:
            return None
        return (float(self.x), float(self.y), float(self.z))


def _parse_triplet(values: list[str], *, kind: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{kind} must have exactly 3 values, got {len(values)}")
    return (float(values[0]), float(values[1]), float(values[2]))


def _parse_triplet_list(
    values: list[str], *, kind: str
) -> list[tuple[float, float, float]]:
    if len(values) == 0 or (len(values) % 3) != 0:
        raise ValueError(f"{kind} must have 3*N values, got {len(values)}")
    triplets: list[tuple[float, float, float]] = []
    for i in range(0, len(values), 3):
        triplets.append((float(values[i]), float(values[i + 1]), float(values[i + 2])))
    return triplets


def _voxel_to_world_mm(affine: np.ndarray, ijk: tuple[int, int, int]) -> CutCoords:
    xyz = nib.affines.apply_affine(affine, np.array(ijk, dtype=float))  # type: ignore
    return CutCoords(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))


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

    # libx264 (and yuv420p) require even width/height. Matplotlib/nilearn can
    # occasionally produce frames off by 1px due to layout/cropping.
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


def _iter_volume_indices(
    n_vols: int, start: int, stop: int | None, step: int
) -> Iterable[int]:
    stop_ = n_vols if stop is None else min(stop, n_vols)
    if start < 0 or start >= n_vols:
        raise ValueError(f"start must be in [0, {n_vols - 1}], got {start}")
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
    """Return up to max_samples finite values from an array without full masking.

    This is used to estimate robust percentiles efficiently.
    """

    flat = np.asarray(data).ravel()
    n = int(flat.size)
    if n == 0 or max_samples <= 0:
        return np.asarray([], dtype=float)

    # If small enough, just take all finite.
    if n <= max_samples:
        vals = flat[np.isfinite(flat)]
        return vals.astype(float, copy=False)

    collected: list[np.ndarray] = []
    remaining = max_samples
    # Draw more than needed to handle NaNs/Infs.
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


def _estimate_vmin_vmax_percentiles(
    img: nib.spatialimages.SpatialImage,  # type: ignore
    *,
    vol_indices: list[int],
    p_low: float,
    p_high: float,
    intensity_mode: str,
    max_total_samples: int,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """Estimate (vmin, vmax) for plotting using robust percentiles.

    - intensity_mode='global' samples across all selected volumes (4D) or the single volume (3D)
    - intensity_mode='frame' is handled per-frame elsewhere

    Returns (None, None) if it cannot find any finite samples.
    """

    p_low, p_high = _validate_percentiles(p_low, p_high)
    rng = np.random.default_rng(seed)

    if intensity_mode != "global":
        raise ValueError(
            f"Unexpected intensity_mode for global estimate: {intensity_mode}"
        )

    if len(vol_indices) == 0:
        return (None, None)

    # Cap total sampling to bound memory/compute.
    samples_per_vol = max(1, int(max_total_samples // max(len(vol_indices), 1)))

    samples: list[np.ndarray] = []
    for vol_index in vol_indices:
        vol_img = img if len(img.shape) == 3 else image.index_img(img, vol_index)
        data = np.asanyarray(vol_img.dataobj)  # type: ignore
        samples.append(
            _sample_finite_values(data, max_samples=samples_per_vol, rng=rng)
        )

    all_samples = (
        np.concatenate([s for s in samples if s.size > 0], axis=0)
        if samples
        else np.asarray([], dtype=float)
    )
    if all_samples.size == 0:
        return (None, None)

    vmin = float(np.percentile(all_samples, p_low))
    vmax = float(np.percentile(all_samples, p_high))
    return (vmin, vmax)


def _prepare_atlas_label_indices(
    atlas_img: nib.spatialimages.SpatialImage,  # type: ignore
) -> tuple[list[int], list[np.ndarray]]:
    """Return sorted atlas labels and per-label flat indices.

    Assumes atlas_img has background already set to NaN if desired.
    """

    atlas_data = np.asanyarray(atlas_img.dataobj)
    flat = np.asarray(atlas_data).ravel()
    finite = np.isfinite(flat)
    if not np.any(finite):
        return ([], [])

    labels_f = np.unique(flat[finite])
    # Most parcellation atlases use integer labels; cast safely when possible.
    if np.all(np.isclose(labels_f, np.round(labels_f))):
        labels = [int(x) for x in np.round(labels_f).astype(int).tolist()]
    else:
        # Fall back to stringified float labels; keep deterministic order.
        labels = [int(x) for x in labels_f.astype(int).tolist()]

    labels = sorted(set(labels))
    indices_per_label: list[np.ndarray] = []
    for lab in labels:
        indices_per_label.append(np.flatnonzero(flat == float(lab)))
    return (labels, indices_per_label)


def _plot_atlas_stats(
    ax: Axes,
    *,
    vol_img: nib.spatialimages.SpatialImage,  # type: ignore
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

    vol_data = np.asanyarray(vol_img.dataobj)
    vol_flat = np.asarray(vol_data).ravel().astype(float, copy=False)

    bxp_stats: list[dict[str, float]] = []
    means: list[float] = []
    positions = np.arange(1, len(labels) + 1, dtype=float)

    for idx in indices_per_label:
        vals = vol_flat[idx]
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
            whislo = float(np.min(inliers_lo)) if inliers_lo.size else float(np.min(vals))
            whishi = float(np.max(inliers_hi)) if inliers_hi.size else float(np.max(vals))
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

    # Box-whisker summary per label.
    ax.bxp(
        bxp_stats,
        positions=positions,
        widths=0.6,
        showfliers=False,
    )

    # Scatter of (subsampled) voxel values with jitter.
    if max_points_per_label > 0 and point_alpha > 0:
        for pos, idx in zip(positions, indices_per_label, strict=True):
            vals = vol_flat[idx]
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

    # Mean marker.
    ax.plot(positions, means, linestyle="none", marker="o", markersize=3, color="red")

    ax.set_xlim(0.5, float(len(labels)) + 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [str(label_id) for label_id in labels],
        rotation=90 if len(labels) > 15 else 0,
    )
    ax.set_xlabel("Atlas label")
    ax.set_ylabel("Voxel values")
    if title:
        ax.set_title(title)
    if ylim is not None:
        y0, y1 = ylim
        if y0 is not None and y1 is not None:
            ax.set_ylim(float(y0), float(y1))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render NIfTI (3D/4D) to orthogonal plots and optionally encode a movie.",
    )
    p.add_argument(
        "--input", required=True, type=Path, help="Input NIfTI file (.nii/.nii.gz)"
    )
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

    # Plot controls
    p.add_argument(
        "--display-mode",
        default="ortho",
        choices=["ortho", "x", "y", "z", "xz", "yx", "yz"],
        help="Nilearn display mode (default: ortho).",
    )
    p.add_argument(
        "--slice-indices",
        nargs=3,
        metavar=("I", "J", "K"),
        help="Voxel indices (i j k) to cut at (converted to world/mm).",
    )
    p.add_argument(
        "--slice-indices-list",
        nargs="+",
        metavar="N",
        help=(
            "Multiple voxel index triplets (i j k i j k ...), rendered as multiple views in one frame. "
            "Overrides --slice-indices/--cut-coords when provided."
        ),
    )
    p.add_argument(
        "--cut-coords",
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="World coordinates in mm (x y z) to cut at (overrides --slice-indices).",
    )
    p.add_argument(
        "--cut-coords-list",
        nargs="+",
        metavar="N",
        help=(
            "Multiple world-coordinate triplets (x y z x y z ...), rendered as multiple views in one frame. "
            "Overrides --slice-indices/--cut-coords when provided."
        ),
    )
    p.add_argument(
        "--ncols",
        type=int,
        default=None,
        help="Number of columns when rendering multiple views (default: number of views).",
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
            "'global' (default) uses robust percentiles across all selected timepoints; "
            "'frame' recomputes percentiles per timepoint."
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
        "--bg-img",
        type=Path,
        default=None,
        help="Optional background anatomical image for overlay rendering.",
    )
    p.add_argument(
        "--atlas",
        type=Path,
        default=None,
        help=(
            "Optional atlas/parcellation NIfTI (.nii/.nii.gz) used to compute a dynamic "
            "per-label boxplot+scatter subplot per frame. Assumed same space but can be "
            "different resolution (will be resampled)."
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
    p.add_argument("--black-bg", action="store_true", help="Use black background.")
    p.add_argument(
        "--colorbar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show colorbar (default: true).",
    )
    p.add_argument(
        "--annotate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Annotate axes with left/right and coordinates (default: false).",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Optional title template. Use {frame}, {index}, {view}, {time} placeholders.",
    )

    # Time annotation (useful for 4D)
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

    # 4D controls
    p.add_argument(
        "--start", type=int, default=0, help="First volume index (default: 0)."
    )
    p.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Stop volume index (exclusive). Defaults to end.",
    )
    p.add_argument("--step", type=int, default=1, help="Volume step (default: 1).")

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

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if bool(args.time_annotate) and args.tr is None:
        raise ValueError("--tr is required when using --time-annotate")

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    output_path: Path
    if args.output is None:
        # Preserve .nii.gz stems
        stem = input_path.name
        for suffix in (".nii.gz", ".nii"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        output_path = input_path.with_name(f"{stem}.mp4")
    else:
        output_path = args.output

    img = image.load_img(str(input_path))
    ndim = len(img.shape)
    if ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D NIfTI, got shape={img.shape}")

    # Determine cut coords
    # NOTE: If cut_coords is left as None for a 4D image, nilearn will often
    # choose cut positions independently per frame (based on the data), which
    # looks like the movie is "scrolling" through slices. For temporal movies
    # we want fixed slice positions by default.
    views_cut_coords: list[tuple[float, float, float] | None]
    user_provided_cuts = (
        args.cut_coords is not None
        or args.slice_indices is not None
        or args.cut_coords_list is not None
        or args.slice_indices_list is not None
    )

    if args.cut_coords_list is not None:
        triplets = _parse_triplet_list(
            list(map(str, args.cut_coords_list)), kind="--cut-coords-list"
        )
        views_cut_coords = [(float(x), float(y), float(z)) for (x, y, z) in triplets]
    elif args.slice_indices_list is not None:
        if (len(args.slice_indices_list) % 3) != 0:
            raise ValueError(
                f"--slice-indices-list must have 3*N integers, got {len(args.slice_indices_list)}"
            )
        triplets_int: list[tuple[int, int, int]] = []
        for i0 in range(0, len(args.slice_indices_list), 3):
            triplets_int.append(
                (
                    int(args.slice_indices_list[i0]),
                    int(args.slice_indices_list[i0 + 1]),
                    int(args.slice_indices_list[i0 + 2]),
                )
            )
        views_cut_coords = [
            _voxel_to_world_mm(np.asarray(img.affine), ijk).as_tuple_or_none()
            for ijk in triplets_int
        ]
    elif args.cut_coords is not None:
        x, y, z = _parse_triplet(list(map(str, args.cut_coords)), kind="--cut-coords")
        views_cut_coords = [(x, y, z)]
    elif args.slice_indices is not None:
        i, j, k = (
            int(args.slice_indices[0]),
            int(args.slice_indices[1]),
            int(args.slice_indices[2]),
        )
        coords = _voxel_to_world_mm(np.asarray(img.affine), (i, j, k))
        views_cut_coords = [coords.as_tuple_or_none()]
    else:
        views_cut_coords = [None]

    width_px, height_px = int(args.size[0]), int(args.size[1])
    dpi = int(args.dpi)
    figsize = (width_px / dpi, height_px / dpi)

    # Prepare output frames directory
    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.frames_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="nifti_frames_")
        frames_dir = Path(temp_dir_obj.name)
    else:
        frames_dir = args.frames_dir
        frames_dir.mkdir(parents=True, exist_ok=True)

    # If the user asked for frames only and didn't provide a frames dir,
    # keep the temp dir so the result is usable.
    keep_temp_frames = bool(args.keep_frames) or bool(args.no_video)

    try:
        if ndim == 3:
            vol_indices = [0]
        else:
            n_vols = img.shape[3]
            vol_indices = list(
                _iter_volume_indices(n_vols, args.start, args.stop, args.step)
            )

        # If no cut coordinates were provided for a 4D image, choose them once
        # from the first rendered volume and keep fixed across time.
        if (
            ndim == 4
            and (not user_provided_cuts)
            and views_cut_coords == [None]
            and args.display_mode == "ortho"
            and len(vol_indices) > 0
        ):
            ref_img = image.index_img(img, vol_indices[0])
            x, y, z = plotting.find_xyz_cut_coords(ref_img)
            views_cut_coords = [(float(x), float(y), float(z))]

        # Optional atlas for per-label stats (resample once to match plotting grid).
        atlas_img_resampled = None
        atlas_labels: list[int] = []
        atlas_indices_per_label: list[np.ndarray] = []
        if args.atlas is not None:
            if not Path(args.atlas).exists():
                raise FileNotFoundError(str(args.atlas))

            atlas_img = nib_load(str(args.atlas))
            target_img = img if ndim == 3 else image.index_img(img, vol_indices[0])
            atlas_img = image.resample_to_img(
                atlas_img,
                target_img,
                interpolation="nearest",
                force_resample=True,
            )

            if bool(args.atlas_ignore_zero):
                atlas_img_any = cast(Any, atlas_img)
                atlas_data = np.asanyarray(atlas_img_any.get_fdata())
                atlas_data = atlas_data.astype(float, copy=False)
                atlas_data[atlas_data == 0] = np.nan
                atlas_img = Nifti1Image(
                    atlas_data,
                    atlas_img_any.affine,
                    atlas_img_any.header,
                )

            atlas_img_resampled = atlas_img

            atlas_labels, atlas_indices_per_label = _prepare_atlas_label_indices(
                atlas_img_resampled
            )

        # Determine vmin/vmax behavior.
        # - If user provides vmin/vmax explicitly: fixed across frames.
        # - Else: use robust percentiles either globally (default) or per-frame.
        p_low, p_high = _validate_percentiles(
            float(args.auto_percentiles[0]), float(args.auto_percentiles[1])
        )
        global_vmin: float | None = None
        global_vmax: float | None = None
        if args.vmin is None or args.vmax is None:
            if args.intensity_mode == "global":
                global_vmin, global_vmax = _estimate_vmin_vmax_percentiles(
                    img,
                    vol_indices=vol_indices,
                    p_low=p_low,
                    p_high=p_high,
                    intensity_mode="global",
                    max_total_samples=int(args.auto_max_total_samples),
                )

        frame_rng = np.random.default_rng(0)
        stats_rng = np.random.default_rng(0)

        # Render frames
        for frame_number, vol_index in enumerate(vol_indices):
            vol_img = img if ndim == 3 else image.index_img(img, vol_index)

            time_sec: float | None
            if args.tr is None:
                time_sec = None
            else:
                time_sec = (float(args.t0_trs) + float(vol_index)) * float(args.tr)

            if args.vmin is not None and args.vmax is not None:
                frame_vmin = float(args.vmin)
                frame_vmax = float(args.vmax)
            else:
                # Fill in any missing bounds using auto scaling.
                if args.intensity_mode == "frame":
                    data = np.asanyarray(vol_img.dataobj)  # type: ignore
                    samples = _sample_finite_values(
                        data,
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

                frame_vmin = float(args.vmin) if args.vmin is not None else auto_vmin
                frame_vmax = float(args.vmax) if args.vmax is not None else auto_vmax

            n_views = len(views_cut_coords)
            if n_views <= 0:
                raise RuntimeError("No views to render")
            ncols = int(args.ncols) if args.ncols is not None else n_views
            if ncols <= 0:
                raise ValueError(f"--ncols must be positive, got {ncols}")

            nrows_maps = int(np.ceil(n_views / ncols))
            show_stats = atlas_img_resampled is not None

            # Layout: map panels in a grid, with an optional stats axis spanning the bottom.
            fig = plt.figure(figsize=figsize)
            if show_stats:
                height_ratios = [1.0] * nrows_maps + [0.75]
                gs = fig.add_gridspec(
                    nrows=nrows_maps + 1,
                    ncols=ncols,
                    height_ratios=height_ratios,
                )
            else:
                gs = fig.add_gridspec(nrows=nrows_maps, ncols=ncols)

            axes_flat: list[Axes] = []
            for r in range(nrows_maps):
                for c in range(ncols):
                    axes_flat.append(fig.add_subplot(gs[r, c]))

            ax_stats = fig.add_subplot(gs[-1, :]) if show_stats else None

            displays = []
            for view_idx, cut_coords in enumerate(views_cut_coords):
                ax = axes_flat[view_idx]

                title = None
                if args.title:
                    title = str(args.title).format(
                        frame=frame_number,
                        index=vol_index,
                        view=view_idx,
                        time=time_sec,
                    )

                show_colorbar = bool(args.colorbar) and (view_idx == 0)
                display = plotting.plot_img(
                    vol_img,
                    display_mode=args.display_mode,
                    cut_coords=cut_coords,
                    cmap=args.cmap,
                    vmin=frame_vmin,
                    vmax=frame_vmax,
                    black_bg=args.black_bg,
                    bg_img=str(args.bg_img) if args.bg_img is not None else None,
                    colorbar=show_colorbar,
                    annotate=bool(args.annotate),
                    title=title,
                    figure=fig,
                    axes=ax,
                )
                displays.append(display)

            if ax_stats is not None:
                stats_title = None
                if time_sec is not None:
                    stats_title = f"Atlas voxel distributions (t={time_sec:.3f}s)"
                elif ndim == 4:
                    stats_title = f"Atlas voxel distributions (index={vol_index})"
                _plot_atlas_stats(
                    ax_stats,
                    vol_img=vol_img,
                    labels=atlas_labels,
                    indices_per_label=atlas_indices_per_label,
                    rng=stats_rng,
                    max_points_per_label=int(args.atlas_stats_max_points_per_label),
                    point_alpha=float(args.atlas_stats_point_alpha),
                    jitter=float(args.atlas_stats_jitter),
                    title=stats_title,
                    ylim=(frame_vmin, frame_vmax)
                    if (frame_vmin is not None and frame_vmax is not None)
                    else None,
                )

            # Turn off any unused axes
            for ax in axes_flat[n_views:]:
                ax.axis("off")

            fig.tight_layout()

            if bool(args.time_annotate):
                label = "t = ? s" if time_sec is None else f"t = {time_sec:.3f} s"
                color = "white" if bool(args.black_bg) else "black"
                fig.text(0.01, 0.99, label, ha="left", va="top", color=color)

            frame_path = frames_dir / f"frame_{frame_number:05d}.png"
            fig.savefig(str(frame_path), dpi=dpi)
            for d in displays:
                try:
                    d.close()  # type: ignore
                except Exception:
                    pass
            plt.close(fig)

        if args.no_video:
            print(f"Rendered {len(vol_indices)} frame(s) to: {frames_dir}")
            return 0

        ffmpeg = _find_ffmpeg(args.ffmpeg)
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg not found. Install system ffmpeg, or add optional extra 'viz' "
                "(imageio-ffmpeg) and re-run. You can also use --no-video to only render frames."
            )

        # Build input glob/pattern for ffmpeg. It expects printf-style numbering.
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
