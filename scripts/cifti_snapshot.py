"""Render a static snapshot of a CIFTI-2 dense timeseries (dtseries) as a PNG.

This is the static companion to scripts/cifti_movie.py and mirrors the CLI style
of scripts/nifti_snapshot.py where applicable.

- Input: dense CIFTI dtseries (.dtseries.nii) or scalar (.dscalar.nii)
- Output: a single PNG for a selected frame/index
- Rendering: fsLR32k inflated surfaces (L/R) via nilearn surface plotting
- Optional atlas: dense label CIFTI (.dlabel.nii) overlayed on the surface maps

Examples
--------
# Basic snapshot (index 0)
uv run python scripts/cifti_snapshot.py --input sub-01_task-rest.dtseries.nii --output snap.png

# Specific timepoint
uv run python scripts/cifti_snapshot.py --input sub-01_task-rest.dtseries.nii --index 30 --output snap30.png

# Multiple views (each view renders both hemispheres)
uv run python scripts/cifti_snapshot.py --input sub-01_task-rest.dtseries.nii --surf-views lateral medial --output snap2views.png

# Atlas overlay
uv run python scripts/cifti_snapshot.py --input sub-01_task-rest.dtseries.nii --index 30 --atlas atlas/TY7.dlabel.nii --output snap30_atlas.png

# Atlas-only snapshot (no input dtseries)
uv run python scripts/cifti_snapshot.py --atlas atlas/TY7.dlabel.nii --output atlas_only.png

# Atlas legend (reads label names/colors from the dlabel)
uv run python scripts/cifti_snapshot.py --atlas atlas/TY7.dlabel.nii --atlas-legend --output atlas_only_legend.png
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any, cast

import matplotlib

# Force a non-interactive backend for speed and headless runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import nibabel as nib
from nibabel.loadsave import load as nib_load
import numpy as np
from nilearn import plotting, surface


@dataclasses.dataclass(frozen=True)
class _CiftiFrameSpec:
    frame_axis_first: bool
    n_frames: int


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
    """Return (frame_axis, brain_axis, frame_spec)."""

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


def _apply_surf_zoom(ax: Axes, zoom: float) -> None:
    try:
        z = float(zoom)
    except Exception:
        return

    if not np.isfinite(z) or z <= 0:
        return

    try:
        base = 10.0
        new_dist = max(1.0, base / z)
        setattr(cast(Any, ax), "dist", new_dist)
    except Exception:
        return


def _add_colorbar(
    *, fig: Figure, cax: Axes, cmap: str, vmin: float, vmax: float
) -> None:
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation="vertical")


def _extract_atlas_label_legend(
    atlas_img: nib.cifti2.cifti2.Cifti2Image,
) -> list[tuple[int, str, tuple[float, float, float, float]]]:
    """Best-effort extraction of (key, name, rgba) from a CIFTI dlabel."""

    def _normalize_rgba(
        rgba4: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        # Some label tables store RGBA in 0–255; Matplotlib expects 0–1.
        mx = max(rgba4)
        if mx > 1.0 and mx <= 255.0:
            return tuple(float(x) / 255.0 for x in rgba4)  # type: ignore[return-value]
        return rgba4

    # 1) Prefer LabelAxis if present
    try:
        from nibabel.cifti2 import cifti2_axes

        ax0 = cifti2_axes.from_index_mapping(atlas_img.header.get_index_map(0))
        ax1 = cifti2_axes.from_index_mapping(atlas_img.header.get_index_map(1))
        label_axis = None
        if isinstance(ax0, cifti2_axes.LabelAxis):
            label_axis = ax0
        elif isinstance(ax1, cifti2_axes.LabelAxis):
            label_axis = ax1
        if label_axis is not None:
            labels_any = cast(Any, getattr(label_axis, "label", None))
            if labels_any is not None and len(labels_any) > 0:
                d = labels_any[0]
                out: list[tuple[int, str, tuple[float, float, float, float]]] = []
                for k, v in cast(Any, d).items():
                    try:
                        name, rgba = v
                    except Exception:
                        continue
                    try:
                        rgba4 = tuple(float(x) for x in rgba)
                        if len(rgba4) != 4:
                            continue
                    except Exception:
                        continue
                    out.append((int(k), str(name), _normalize_rgba(cast(Any, rgba4))))
                out.sort(key=lambda x: x[0])
                return out
    except Exception:
        pass

    # 2) Fallback: named map label table on the matrix indices map (common in dlabel)
    try:
        idx_map0 = atlas_img.header.get_index_map(0)
        named_maps = cast(Any, getattr(idx_map0, "named_maps", None))
        if named_maps is not None and len(named_maps) > 0:
            nm0 = named_maps[0]
            lt = cast(Any, getattr(nm0, "label_table", None))
            if lt is not None:
                out2: list[tuple[int, str, tuple[float, float, float, float]]] = []
                for k, entry in cast(Any, lt).items():
                    name = getattr(entry, "label", None)
                    if name is None:
                        name = getattr(entry, "name", None)
                    if name is None:
                        name = str(entry)
                    rgba = getattr(entry, "rgba", None)
                    if rgba is None:
                        continue
                    rgba4 = tuple(float(x) for x in rgba)
                    if len(rgba4) != 4:
                        continue
                    out2.append((int(k), str(name), _normalize_rgba(cast(Any, rgba4))))
                out2.sort(key=lambda x: x[0])
                return out2
    except Exception:
        pass

    return []


def _draw_atlas_legend(
    ax: Axes,
    labels: list[tuple[int, str, tuple[float, float, float, float]]],
    *,
    ignore_zero: bool,
    max_items: int,
) -> None:
    ax.set_axis_off()
    items: list[tuple[int, str, tuple[float, float, float, float]]] = []
    for k, name, rgba in labels:
        if ignore_zero and int(k) == 0:
            continue
        items.append((int(k), str(name), rgba))

    if max_items > 0:
        items = items[: int(max_items)]

    if not items:
        return

    handles = [
        Patch(facecolor=rgba, edgecolor="none", label=f"{k}: {name}")
        for k, name, rgba in items
    ]
    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        fontsize=6,
        handlelength=1.0,
        handleheight=1.0,
        labelspacing=0.4,
        borderaxespad=0.0,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render a static snapshot PNG from a CIFTI dtseries/dscalar.",
    )

    p.add_argument(
        "--input",
        required=False,
        type=Path,
        default=None,
        help=(
            "Optional input CIFTI dtseries/dscalar. If omitted, you must provide --atlas and an atlas-only snapshot will be rendered."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Defaults to <input stem>_idx-<index>.png (if --input provided) "
            "or <atlas stem>_atlas.png (if atlas-only)."
        ),
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="Frame index to display (default: 0).",
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
            "'global' (default) uses robust percentiles across all timepoints; "
            "'frame' uses percentiles from the selected frame only."
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
        help="Max random samples when estimating frame percentiles (default: 200000).",
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
        "--black-bg", action="store_true", help="Use black figure background."
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
            "Optional title template. Use {index}, {panel}, {view}, {hemi}, {time} placeholders."
        ),
    )

    # Optional atlas overlay
    p.add_argument(
        "--atlas",
        type=Path,
        default=None,
        help=(
            "Optional CIFTI dense label atlas (.dlabel.nii) to overlay on the surface maps."
        ),
    )
    p.add_argument(
        "--atlas-alpha",
        type=float,
        default=0.12,
        help="Alpha for atlas overlay (default: 0.12).",
    )
    p.add_argument(
        "--atlas-cmap",
        type=str,
        default="tab20",
        help="Colormap for atlas overlay (default: tab20).",
    )
    p.add_argument(
        "--atlas-ignore-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat atlas label 0 as background (default: true).",
    )
    p.add_argument(
        "--atlas-legend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render a legend using label names/colors from the atlas (default: false).",
    )
    p.add_argument(
        "--atlas-legend-max-items",
        type=int,
        default=40,
        help="Max legend entries to show (default: 40).",
    )

    # Optional time annotation (for convenience; matches cifti_movie.py)
    p.add_argument(
        "--time-annotate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Annotate snapshot with time in seconds (default: false).",
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

    # Output resolution
    p.add_argument(
        "--dpi", type=int, default=150, help="DPI for saved PNG (default: 150)."
    )
    p.add_argument(
        "--size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(1280, 720),
        help="Figure size in pixels (default: 1280 720).",
    )

    return p


def _derive_default_output(input_path: Path, *, index: int) -> Path:
    stem = input_path.name
    for suffix in (".dtseries.nii", ".dscalar.nii", ".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return input_path.with_name(f"{stem}_idx-{index}.png")


def _derive_default_output_atlas_only(atlas_path: Path) -> Path:
    stem = atlas_path.name
    for suffix in (".dlabel.nii", ".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return atlas_path.with_name(f"{stem}_atlas.png")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    input_path: Path | None = args.input
    atlas_path: Path | None = args.atlas

    if input_path is None and atlas_path is None:
        raise ValueError("You must provide at least one of --input or --atlas")

    if input_path is not None and not input_path.exists():
        raise FileNotFoundError(str(input_path))

    img: nib.cifti2.cifti2.Cifti2Image | None = None
    structures: dict[str, tuple[slice, Any]] | None = None
    cortex_mask: np.ndarray | None = None
    frame_spec: _CiftiFrameSpec | None = None
    n_frames: int | None = None
    if input_path is not None:
        loaded = nib_load(str(input_path))
        if not isinstance(loaded, nib.cifti2.cifti2.Cifti2Image):
            raise TypeError(f"Expected CIFTI-2 image, got {type(loaded)}")
        img = loaded

        _frame_axis, brain_axis, inferred_spec = _infer_cifti_axes(img)
        frame_spec = inferred_spec
        structures = _extract_cortex_structures(brain_axis)
        cortex_mask = _compute_cortex_mask(brain_axis, structures)

        n_frames = int(frame_spec.n_frames)
        if n_frames <= 0:
            raise ValueError("No frames found")
        if int(args.index) < 0 or int(args.index) >= n_frames:
            raise ValueError(
                f"--index must be in [0, {n_frames - 1}], got {int(args.index)}"
            )

    # Load surfaces once.
    surf_left_mesh = surface.load_surf_mesh(str(args.surf_left))
    surf_right_mesh = surface.load_surf_mesh(str(args.surf_right))

    # Optional atlas overlay (vertex-space for each hemi)
    atlas_left: np.ndarray | None = None
    atlas_right: np.ndarray | None = None
    atlas_left_plot: np.ndarray | None = None
    atlas_right_plot: np.ndarray | None = None
    atlas_cmap_obj = None
    atlas_labels: list[tuple[int, str, tuple[float, float, float, float]]] = []
    if atlas_path is not None:
        if not atlas_path.exists():
            raise FileNotFoundError(str(atlas_path))
        atlas_img = nib_load(str(atlas_path))
        if not isinstance(atlas_img, nib.cifti2.cifti2.Cifti2Image):
            raise TypeError(f"Expected atlas CIFTI-2 image, got {type(atlas_img)}")
        _atlas_frame_axis, atlas_brain, atlas_spec = _infer_cifti_axes(atlas_img)
        atlas_vec = _get_frame_vector(atlas_img, frame_index=0, frame_spec=atlas_spec)
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

        if bool(args.atlas_ignore_zero):
            atlas_left = np.asarray(atlas_left, dtype=float, copy=True)
            atlas_right = np.asarray(atlas_right, dtype=float, copy=True)
            atlas_left[atlas_left == 0] = np.nan
            atlas_right[atlas_right == 0] = np.nan

        if bool(args.atlas_legend):
            atlas_labels = _extract_atlas_label_legend(atlas_img)

        # Decide how to color the atlas overlay.
        # - Default: use user-provided --atlas-cmap (e.g. tab20)
        # - If --atlas-legend: use the atlas' own label-table RGBA so legend and overlay match.
        if bool(args.atlas_legend) and len(atlas_labels) > 0:
            entries = atlas_labels
            if bool(args.atlas_ignore_zero):
                entries = [e for e in entries if int(e[0]) != 0]

            keys = np.array([int(k) for k, _name, _rgba in entries], dtype=np.int64)
            colors = [rgba for _k, _name, rgba in entries]
            if keys.size > 0:
                order = np.argsort(keys)
                keys = keys[order]
                colors = [colors[int(i)] for i in order]

            atlas_cmap_obj = ListedColormap(colors, name="atlas_label_table")
            try:
                cast(Any, atlas_cmap_obj).set_bad((0.0, 0.0, 0.0, 0.0))
            except Exception:
                pass

            def _remap_labels(arr: np.ndarray) -> np.ndarray:
                a = np.asarray(arr, dtype=float)
                out = np.full(a.shape, np.nan, dtype=float)
                m = np.isfinite(a)
                if not np.any(m) or keys.size == 0:
                    return out
                vals = a[m].astype(np.int64, copy=False)
                idx = np.searchsorted(keys, vals)
                ok = (idx >= 0) & (idx < keys.size) & (keys[idx] == vals)
                tmp = np.full(vals.shape, np.nan, dtype=float)
                tmp[ok] = idx[ok].astype(float)
                out[m] = tmp
                return out

            atlas_left_plot = _remap_labels(atlas_left)
            atlas_right_plot = _remap_labels(atlas_right)
        else:
            # Make background (NaNs / masked) fully transparent so it doesn't look like a network.
            # This is especially important for colormaps like tab20 that include gray tones.
            atlas_cmap_obj = plt.get_cmap(str(args.atlas_cmap))
            try:
                atlas_cmap_obj = cast(Any, atlas_cmap_obj).copy()
            except Exception:
                pass
            try:
                cast(Any, atlas_cmap_obj).set_bad((0.0, 0.0, 0.0, 0.0))
            except Exception:
                pass
            atlas_left_plot = atlas_left
            atlas_right_plot = atlas_right

    # Determine vmin/vmax
    p_low, p_high = _validate_percentiles(
        float(args.auto_percentiles[0]), float(args.auto_percentiles[1])
    )

    vmin: float | None
    vmax: float | None
    if img is None:
        vmin, vmax = None, None
    elif args.vmin is not None and args.vmax is not None:
        vmin, vmax = float(args.vmin), float(args.vmax)
    else:
        assert frame_spec is not None
        assert n_frames is not None
        assert cortex_mask is not None

        if str(args.intensity_mode) == "global" and n_frames > 1:
            rng = np.random.default_rng(0)
            max_total = int(args.auto_max_total_samples)
            samples_per_frame = max(1, int(max_total // max(n_frames, 1)))
            collected: list[np.ndarray] = []
            for fi in range(n_frames):
                vec = _get_frame_vector(img, frame_index=int(fi), frame_spec=frame_spec)
                vec = vec[cortex_mask]
                collected.append(
                    _sample_finite_values(vec, max_samples=samples_per_frame, rng=rng)
                )
            all_samples = (
                np.concatenate([s for s in collected if s.size > 0], axis=0)
                if collected
                else np.asarray([], dtype=float)
            )
            if all_samples.size > 0:
                vmin = float(np.percentile(all_samples, p_low))
                vmax = float(np.percentile(all_samples, p_high))
            else:
                vmin = None
                vmax = None
        else:
            rng = np.random.default_rng(0)
            vec = _get_frame_vector(
                img, frame_index=int(args.index), frame_spec=frame_spec
            )
            vec = vec[cortex_mask]
            samples = _sample_finite_values(
                vec, max_samples=int(args.auto_max_samples), rng=rng
            )
            if samples.size > 0:
                vmin = float(np.percentile(samples, p_low))
                vmax = float(np.percentile(samples, p_high))
            else:
                vmin = None
                vmax = None

        if args.vmin is not None:
            vmin = float(args.vmin)
        if args.vmax is not None:
            vmax = float(args.vmax)

    # Prepare figure
    width_px, height_px = int(args.size[0]), int(args.size[1])
    dpi = int(args.dpi)
    figsize = (width_px / dpi, height_px / dpi)

    views = [str(v) for v in args.surf_views]
    panels: list[tuple[str, str]] = []
    for view in views:
        panels.append((view, "L"))
        panels.append((view, "R"))

    n_panels = len(panels)
    if n_panels <= 0:
        raise RuntimeError("No surface panels to render")

    ncols = int(args.ncols) if args.ncols is not None else int(n_panels)
    if ncols <= 0:
        raise ValueError(f"--ncols must be positive, got {ncols}")
    nrows_maps = int(np.ceil(n_panels / ncols))

    want_cbar = bool(args.colorbar) and (vmin is not None and vmax is not None)
    want_legend = (
        bool(args.atlas_legend) and (atlas_path is not None) and (len(atlas_labels) > 0)
    )
    side = str(args.colorbar_side)

    want_side_col = bool(want_cbar or want_legend)
    ncols_total = int(ncols + (1 if want_side_col else 0))
    surf_col0 = 1 if (want_side_col and side == "left") else 0
    side_col = 0 if (want_side_col and side == "left") else (ncols_total - 1)

    side_width_ratio = 0.08
    if want_legend and want_cbar:
        side_width_ratio = 0.18
    elif want_legend and not want_cbar:
        side_width_ratio = 0.22

    width_ratios = None
    if want_side_col:
        if side == "left":
            width_ratios = [side_width_ratio] + [1.0] * ncols
        else:
            width_ratios = [1.0] * ncols + [side_width_ratio]

    fig = plt.figure(figsize=figsize)
    if bool(args.black_bg):
        fig.patch.set_facecolor("black")

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
            axes_flat.append(fig.add_subplot(gs[r, c + surf_col0], projection="3d"))

    ax_cbar = None
    ax_legend = None
    if want_side_col:
        ax_side_container = fig.add_subplot(gs[:nrows_maps, side_col])
        ax_side_container.set_axis_off()
        if want_cbar:
            ax_cbar = inset_axes(
                ax_side_container,
                width="55%",
                height=("55%" if want_legend else "70%"),
                loc="upper center" if want_legend else "center",
                borderpad=0.0,
            )
        if want_legend:
            ax_legend = inset_axes(
                ax_side_container,
                width="100%",
                height=("40%" if want_cbar else "90%"),
                loc="lower center" if want_cbar else "center",
                borderpad=0.0,
            )

    # Extract selected frame (if present)
    left_map: np.ndarray | None = None
    right_map: np.ndarray | None = None
    if img is not None:
        assert frame_spec is not None
        assert structures is not None
        frame_vec = _get_frame_vector(
            img, frame_index=int(args.index), frame_spec=frame_spec
        )
        left_map = _brain_to_hemi_vertices(
            frame_vec=frame_vec,
            structures=structures,
            structure_name="CIFTI_STRUCTURE_CORTEX_LEFT",
        )
        right_map = _brain_to_hemi_vertices(
            frame_vec=frame_vec,
            structures=structures,
            structure_name="CIFTI_STRUCTURE_CORTEX_RIGHT",
        )

    time_s: float | None = None
    if bool(args.time_annotate):
        if img is None:
            raise ValueError("--time-annotate requires --input")
        if args.tr is None:
            raise ValueError("--tr is required when --time-annotate is true")
        time_s = (float(args.t0_trs) + float(args.index)) * float(args.tr)
        color = "white" if bool(args.black_bg) else "black"
        fig.text(0.01, 0.99, f"t={time_s:.2f}s", ha="left", va="top", color=color)

    # Plot panels
    for panel_idx, (view, hemi) in enumerate(panels):
        if panel_idx >= len(axes_flat):
            break
        ax = axes_flat[panel_idx]
        surf_mesh = surf_left_mesh if hemi == "L" else surf_right_mesh
        stat_map = left_map if hemi == "L" else right_map

        title = None
        if args.title:
            title = str(args.title).format(
                index=int(args.index),
                panel=int(panel_idx),
                view=str(view),
                hemi=str(hemi),
                time=("" if time_s is None else f"{time_s:.3f}"),
            )

        if left_map is not None and right_map is not None:
            stat_map = left_map if hemi == "L" else right_map
            plotting.plot_surf_stat_map(
                surf_mesh,
                stat_map,
                hemi="left" if hemi == "L" else "right",
                view=str(view),
                cmap=str(args.cmap),
                vmin=vmin,
                vmax=vmax,
                colorbar=False,
                title=title,
                figure=fig,
                axes=ax,
            )
        elif title is not None:
            ax.set_title(title)

        if atlas_left_plot is not None and atlas_right_plot is not None:
            roi_map = atlas_left_plot if hemi == "L" else atlas_right_plot
            roi_map = np.ma.masked_invalid(roi_map)
            plotting.plot_surf_roi(
                surf_mesh,
                roi_map,
                hemi="left" if hemi == "L" else "right",
                view=str(view),
                cmap=(
                    cast(Any, atlas_cmap_obj)
                    if atlas_cmap_obj is not None
                    else str(args.atlas_cmap)
                ),
                alpha=float(args.atlas_alpha),
                colorbar=False,
                figure=fig,
                axes=ax,
            )

        try:
            for child in ax.get_children():
                if child.__class__.__name__ == "Poly3DCollection":
                    try:
                        child_any = cast(Any, child)
                        child_any.set_edgecolor("none")
                        child_any.set_linewidth(0)
                    except Exception:
                        pass
        except Exception:
            pass

        _apply_surf_zoom(ax, float(args.surf_zoom))

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    if ax_cbar is not None and vmin is not None and vmax is not None:
        _add_colorbar(
            fig=fig,
            cax=ax_cbar,
            cmap=str(args.cmap),
            vmin=float(vmin),
            vmax=float(vmax),
        )

    if ax_legend is not None and atlas_labels:
        _draw_atlas_legend(
            ax_legend,
            atlas_labels,
            ignore_zero=bool(args.atlas_ignore_zero),
            max_items=int(args.atlas_legend_max_items),
        )

    output_path: Path
    if args.output is not None:
        output_path = Path(args.output)
    elif input_path is not None:
        output_path = _derive_default_output(input_path, index=int(args.index))
    else:
        assert atlas_path is not None
        output_path = _derive_default_output_atlas_only(atlas_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.95)
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)

    print(f"Wrote snapshot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
