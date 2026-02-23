"""Render a static snapshot of a NIfTI image (3D/4D) as a PNG.

This is the static companion to scripts/nifti_movie.py. It supports:
- orthogonal views by default (sag/cor/ax)
- fixed cuts in world coords or voxel indices
- multiple cut locations rendered as multiple panels in a grid
- optional atlas/parcellation overlay (resampled if needed)
- robust intensity scaling (global across time for 4D by default)

Examples
--------
# 3D snapshot (auto cut coords)
python scripts/nifti_snapshot.py --input stat.nii.gz --output stat.png

# 4D snapshot (defaults to index 0)
python scripts/nifti_snapshot.py --input bold.nii.gz --output bold0.png

# 4D snapshot at a specific timepoint
python scripts/nifti_snapshot.py --input bold.nii.gz --index 30 --output bold30.png

# Two cut locations, stacked vertically
python scripts/nifti_snapshot.py --input bold.nii.gz --index 30 --output bold30_2views.png \
  --cut-coords-list 0 -25 50  0 -10 40 --ncols 1

# Atlas overlay
python scripts/nifti_snapshot.py --input bold.nii.gz --index 30 --output bold30_atlas.png \
  --atlas atlas_parcellation.nii.gz --atlas-alpha 0.12 --atlas-cmap tab20
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
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


def _estimate_vmin_vmax_percentiles_4d_global(
    img: nib.spatialimages.SpatialImage,  # type: ignore
    *,
    p_low: float,
    p_high: float,
    max_total_samples: int,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """Estimate (vmin, vmax) using robust percentiles across all timepoints.

    Uses random sampling to keep runtime bounded.
    """

    p_low, p_high = _validate_percentiles(p_low, p_high)
    rng = np.random.default_rng(seed)

    if len(img.shape) != 4:
        data = np.asanyarray(img.dataobj)  # type: ignore
        sample_vals = _sample_finite_values(
            data, max_samples=max_total_samples, rng=rng
        )
        if sample_vals.size == 0:
            return (None, None)
        return (
            float(np.percentile(sample_vals, p_low)),
            float(np.percentile(sample_vals, p_high)),
        )

    n_vols = int(img.shape[3])
    if n_vols <= 0:
        return (None, None)

    samples_per_vol = max(1, int(max_total_samples // max(n_vols, 1)))
    samples_list: list[np.ndarray] = []
    for vol_index in range(n_vols):
        vol_img = image.index_img(img, vol_index)
        data = np.asanyarray(vol_img.dataobj)  # type: ignore
        samples_list.append(
            _sample_finite_values(data, max_samples=samples_per_vol, rng=rng)
        )

    all_samples = (
        np.concatenate([s for s in samples_list if s.size > 0], axis=0)
        if samples_list
        else np.asarray([], dtype=float)
    )
    if all_samples.size == 0:
        return (None, None)

    return (
        float(np.percentile(all_samples, p_low)),
        float(np.percentile(all_samples, p_high)),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render a static snapshot PNG from a NIfTI (3D/4D).",
    )

    p.add_argument(
        "--input",
        required=False,
        type=Path,
        default=None,
        help="Optional base NIfTI (.nii/.nii.gz). If omitted, renders the atlas as the base image.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <input stem>_idx-<index>.png (4D) or <stem>.png (3D).",
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="If input is 4D, which volume index to display (default: 0).",
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
            "Multiple voxel index triplets (i j k i j k ...), rendered as multiple views in one image. "
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
            "Multiple world-coordinate triplets (x y z x y z ...), rendered as multiple views in one image. "
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
            "'global' (default) uses robust percentiles across all timepoints for 4D; "
            "'frame' uses percentiles from the selected volume only."
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
            "Optional atlas/parcellation NIfTI (.nii/.nii.gz) to overlay. "
            "Assumed same space but can be different resolution (will be resampled)."
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
        help=(
            "Render a legend for the atlas labels (default: false). "
            "Note: NIfTI files typically do not embed label names/colors; use --atlas-labels to provide them."
        ),
    )
    p.add_argument(
        "--atlas-labels",
        type=Path,
        default=None,
        help=(
            "Optional TSV/JSON file mapping atlas label values to names and/or colors for the legend. "
            "TSV columns: id,name[,r,g,b,a] or id,name,hex. JSON: dict id->name or list of {id,name,color}."
        ),
    )
    p.add_argument(
        "--atlas-legend-max-items",
        type=int,
        default=40,
        help="Max legend entries to show (default: 40).",
    )
    p.add_argument(
        "--atlas-legend-fontsize",
        type=float,
        default=6.0,
        help="Legend font size (default: 6).",
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
        help="Optional title template. Use {index} and {view} placeholders.",
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


def _derive_default_output(input_path: Path, *, is_4d: bool, index: int) -> Path:
    stem = input_path.name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if is_4d:
        return input_path.with_name(f"{stem}_idx-{index}.png")
    return input_path.with_name(f"{stem}.png")


def _normalize_rgba(
    rgba: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    mx = max(rgba)
    if mx > 1.0 and mx <= 255.0:
        return (rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0, rgba[3] / 255.0)
    return rgba


def _hex_to_rgba(s: str) -> tuple[float, float, float, float] | None:
    t = s.strip()
    if t.startswith("#"):
        t = t[1:]
    if len(t) not in (6, 8):
        return None
    try:
        r = int(t[0:2], 16)
        g = int(t[2:4], 16)
        b = int(t[4:6], 16)
        a = int(t[6:8], 16) if len(t) == 8 else 255
    except Exception:
        return None
    return _normalize_rgba((float(r), float(g), float(b), float(a)))


def _load_atlas_label_map(
    path: Path,
) -> dict[int, tuple[str | None, tuple[float, float, float, float] | None]]:
    """Load a best-effort label mapping.

    Returns: id -> (name, rgba)
    """

    txt = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        obj = json.loads(txt)
        out: dict[int, tuple[str | None, tuple[float, float, float, float] | None]] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    kid = int(k)
                except Exception:
                    continue
                if isinstance(v, str):
                    out[kid] = (v, None)
                elif isinstance(v, dict):
                    name = v.get("name") if isinstance(v.get("name"), str) else None
                    color = v.get("color")
                    rgba: tuple[float, float, float, float] | None = None
                    if isinstance(color, str):
                        rgba = _hex_to_rgba(color)
                    elif isinstance(color, (list, tuple)) and len(color) in (3, 4):
                        try:
                            r = float(color[0])
                            g = float(color[1])
                            b = float(color[2])
                            a = float(color[3]) if len(color) == 4 else 1.0
                            rgba = _normalize_rgba((r, g, b, a))
                        except Exception:
                            rgba = None
                    out[kid] = (name, rgba)
            return out

        if isinstance(obj, list):
            for item in obj:
                if not isinstance(item, dict) or "id" not in item:
                    continue
                try:
                    kid = int(item["id"])
                except Exception:
                    continue
                name = item.get("name") if isinstance(item.get("name"), str) else None
                color = item.get("color")
                rgba2: tuple[float, float, float, float] | None = None
                if isinstance(color, str):
                    rgba2 = _hex_to_rgba(color)
                elif isinstance(color, (list, tuple)) and len(color) in (3, 4):
                    try:
                        r = float(color[0])
                        g = float(color[1])
                        b = float(color[2])
                        a = float(color[3]) if len(color) == 4 else 1.0
                        rgba2 = _normalize_rgba((r, g, b, a))
                    except Exception:
                        rgba2 = None
                out[kid] = (name, rgba2)
            return out

        return out

    # TSV/CSV-ish (tab preferred). Lines: id, name, (optional) hex or r g b (a)
    out2: dict[int, tuple[str | None, tuple[float, float, float, float] | None]] = {}
    for raw in txt.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [
            p.strip() for p in line.replace(",", "\t").split("\t") if p.strip() != ""
        ]
        if len(parts) < 2:
            continue
        try:
            kid2 = int(parts[0])
        except Exception:
            continue
        name2 = parts[1]
        rgba3: tuple[float, float, float, float] | None = None
        if len(parts) >= 3:
            maybe_hex = _hex_to_rgba(parts[2])
            if maybe_hex is not None:
                rgba3 = maybe_hex
            elif len(parts) >= 5:
                try:
                    r = float(parts[2])
                    g = float(parts[3])
                    b = float(parts[4])
                    a = float(parts[5]) if len(parts) >= 6 else 1.0
                    rgba3 = _normalize_rgba((r, g, b, a))
                except Exception:
                    rgba3 = None
        out2[kid2] = (name2, rgba3)
    return out2


def _build_discrete_atlas_colormap(
    label_ids: np.ndarray,
    *,
    base_cmap_name: str,
    label_map: dict[int, tuple[str | None, tuple[float, float, float, float] | None]]
    | None,
) -> tuple[ListedColormap, list[tuple[int, str, tuple[float, float, float, float]]]]:
    """Return (cmap, legend_entries).

    legend_entries: (label_id, name, rgba)
    """

    cmap = plt.get_cmap(str(base_cmap_name))
    colors: list[tuple[float, float, float, float]] = []
    entries: list[tuple[int, str, tuple[float, float, float, float]]] = []

    n = int(label_ids.size)
    if n <= 0:
        cm = ListedColormap([], name="atlas_empty")
        try:
            cm.set_bad((0.0, 0.0, 0.0, 0.0))
        except Exception:
            pass
        return cm, []

    if label_map is None:
        label_map = {}

    # Determine colors in order.
    for i, lid in enumerate(label_ids.tolist()):
        lid_int = int(lid)
        name, rgba = label_map.get(lid_int, (None, None))
        if rgba is None:
            # Pick a stable color from the base colormap.
            if getattr(cmap, "N", None) is not None and int(getattr(cmap, "N")) > 0:
                rgba = cast(Any, cmap)(i / max(n - 1, 1))
            else:
                rgba = (0.5, 0.5, 0.5, 1.0)
        rgba4 = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
        rgba4 = _normalize_rgba(rgba4)
        colors.append(rgba4)
        entries.append(
            (lid_int, (name if name is not None else f"Label {lid_int}"), rgba4)
        )

    cm2 = ListedColormap(colors, name="atlas_discrete")
    try:
        cm2.set_bad((0.0, 0.0, 0.0, 0.0))
    except Exception:
        pass
    return cm2, entries


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.input is None and args.atlas is None:
        raise ValueError("At least one of --input or --atlas must be provided")

    base_path = args.input if args.input is not None else args.atlas
    if base_path is None:
        raise RuntimeError("Expected a base path")
    if not base_path.exists():
        raise FileNotFoundError(str(base_path))

    atlas_only_mode = args.input is None and args.atlas is not None
    use_bg_as_base = bool(atlas_only_mode and (args.bg_img is not None))

    # Choose what we render as the base image.
    # - Normal: base is --input
    # - Atlas-only: base is --atlas
    # - Atlas-only + --bg-img: base is --bg-img, atlas becomes an overlay (so --atlas-alpha works)
    base_img_path = args.bg_img if use_bg_as_base else base_path
    if base_img_path is None:
        raise RuntimeError("Expected a base image path")
    if not Path(base_img_path).exists():
        raise FileNotFoundError(str(base_img_path))

    base_img = image.load_img(str(base_img_path))
    ndim = len(base_img.shape)
    if ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D NIfTI, got shape={base_img.shape}")

    if ndim == 4:
        n_vols = int(base_img.shape[3])
        if args.index < 0 or args.index >= n_vols:
            raise ValueError(f"--index must be in [0, {n_vols - 1}], got {args.index}")
        vol_img = image.index_img(base_img, int(args.index))
    else:
        vol_img = base_img

    # If we're rendering atlas-only and want a legend, treat the atlas as a label image.
    # When use_bg_as_base is true, the atlas will be overlaid (not plotted as the base).

    output_path = args.output or _derive_default_output(
        base_path, is_4d=(ndim == 4), index=int(args.index)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Views / cut coords
    if args.cut_coords_list is not None:
        triplets = _parse_triplet_list(
            list(map(str, args.cut_coords_list)), kind="--cut-coords-list"
        )
        views_cut_coords: list[tuple[float, float, float] | None] = [
            (float(x), float(y), float(z)) for (x, y, z) in triplets
        ]
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
            _voxel_to_world_mm(np.asarray(vol_img.affine), ijk).as_tuple_or_none()  # type: ignore
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
        coords = _voxel_to_world_mm(np.asarray(vol_img.affine), (i, j, k))  # type: ignore
        views_cut_coords = [coords.as_tuple_or_none()]
    else:
        # For ortho displays, pick stable cuts once so nilearn doesn't pick per-view.
        # For atlas-only mode, let nilearn choose cut coords inside plot_roi to avoid
        # find_xyz_cut_coords warnings on discrete/NaN-masked label images.
        if atlas_only_mode and (not use_bg_as_base):
            views_cut_coords = [None]
        elif args.display_mode == "ortho":
            x, y, z = plotting.find_xyz_cut_coords(vol_img)
            views_cut_coords = [(float(x), float(y), float(z))]
        else:
            views_cut_coords = [None]

    # Atlas overlay (resample once to match plotting grid)
    # If --input is omitted and bg is not used as base, the atlas is already the base image.
    atlas_img_resampled = None
    atlas_legend_entries: list[tuple[int, str, tuple[float, float, float, float]]] = []
    atlas_cmap_obj: ListedColormap | None = None
    atlas_discrete_vmin_vmax: tuple[float, float] | None = None
    label_map: (
        dict[int, tuple[str | None, tuple[float, float, float, float] | None]] | None
    ) = None
    if bool(args.atlas_legend) and args.atlas_labels is not None:
        if not Path(args.atlas_labels).exists():
            raise FileNotFoundError(str(args.atlas_labels))
        label_map = _load_atlas_label_map(Path(args.atlas_labels))

    atlas_is_overlay = bool(
        args.atlas is not None and (args.input is not None or use_bg_as_base)
    )

    if atlas_is_overlay:
        if not Path(args.atlas).exists():
            raise FileNotFoundError(str(args.atlas))
        atlas_img = cast(Any, image.load_img(str(args.atlas)))
        atlas_img = cast(
            Any,
            image.resample_to_img(
                atlas_img,
                vol_img,
                interpolation="nearest",
                force_resample=True,
            ),
        )
        if bool(args.atlas_ignore_zero):
            atlas_data = np.asanyarray(cast(Any, atlas_img).dataobj)
            atlas_data = atlas_data.astype(float, copy=False)
            atlas_data[atlas_data == 0] = np.nan
            atlas_img = Nifti1Image(
                atlas_data,
                cast(Any, atlas_img).affine,
                cast(Any, atlas_img).header,
            )
        atlas_img_resampled = atlas_img

        if bool(args.atlas_legend):
            atlas_data2 = np.asanyarray(cast(Any, atlas_img).dataobj)
            atlas_vals = atlas_data2[np.isfinite(atlas_data2)]
            if atlas_vals.size > 0:
                atlas_ids = np.unique(atlas_vals.astype(np.int64, copy=False))
                atlas_cmap_obj, atlas_legend_entries = _build_discrete_atlas_colormap(
                    atlas_ids,
                    base_cmap_name=str(args.atlas_cmap),
                    label_map=label_map,
                )

                # We remap atlas values to contiguous indices 1..N (leaving 0 unused).
                # Some plotting paths treat 0 as background; shifting by +1 keeps the
                # first parcel visible while preserving stable, discrete coloring.
                atlas_discrete_vmin_vmax = (1.0, float(int(atlas_ids.size)))

                # Remap atlas values to contiguous indices 1..N so colors match the legend.
                remap = np.full(atlas_data2.shape, np.nan, dtype=float)
                m = np.isfinite(atlas_data2)
                vals_int = atlas_data2[m].astype(np.int64, copy=False)
                idx = np.searchsorted(atlas_ids, vals_int)
                ok = (idx >= 0) & (idx < atlas_ids.size) & (atlas_ids[idx] == vals_int)
                tmp = np.full(vals_int.shape, np.nan, dtype=float)
                tmp[ok] = (idx[ok] + 1).astype(float)
                remap[m] = tmp
                atlas_img_resampled = Nifti1Image(
                    remap,
                    cast(Any, atlas_img).affine,
                    cast(Any, atlas_img).header,
                )

    # Atlas-only base image: optionally remap to discrete indices for legend consistency.
    # Only applies when the atlas itself is the base (i.e., no bg underlay base).
    if atlas_only_mode and (not use_bg_as_base) and bool(args.atlas_legend):
        data0 = np.asanyarray(vol_img.dataobj)  # type: ignore
        data0 = data0.astype(float, copy=False)
        if bool(args.atlas_ignore_zero):
            data0 = np.asarray(data0, dtype=float, copy=True)
            data0[data0 == 0] = np.nan
        vals0 = data0[np.isfinite(data0)]
        if vals0.size > 0:
            ids0 = np.unique(vals0.astype(np.int64, copy=False))
            atlas_cmap_obj, atlas_legend_entries = _build_discrete_atlas_colormap(
                ids0,
                base_cmap_name=str(args.atlas_cmap),
                label_map=label_map,
            )
            atlas_discrete_vmin_vmax = (1.0, float(int(ids0.size)))
            remap0 = np.full(data0.shape, np.nan, dtype=float)
            m0 = np.isfinite(data0)
            v0 = data0[m0].astype(np.int64, copy=False)
            idx0 = np.searchsorted(ids0, v0)
            ok0 = (idx0 >= 0) & (idx0 < ids0.size) & (ids0[idx0] == v0)
            tmp0 = np.full(v0.shape, np.nan, dtype=float)
            tmp0[ok0] = (idx0[ok0] + 1).astype(float)
            remap0[m0] = tmp0
            vol_img = Nifti1Image(
                remap0, cast(Any, vol_img).affine, cast(Any, vol_img).header
            )

    # Atlas-only base image without legend: still honor --atlas-ignore-zero so that
    # background label 0 is not rendered as a giant colored parcel.
    if atlas_only_mode and (not use_bg_as_base) and (not bool(args.atlas_legend)):
        if bool(args.atlas_ignore_zero):
            data_plain = np.asanyarray(vol_img.dataobj)  # type: ignore
            data_plain = data_plain.astype(float, copy=False)
            if np.any(data_plain == 0):
                data_plain = np.asarray(data_plain, dtype=float, copy=True)
                data_plain[data_plain == 0] = np.nan
                vol_img = Nifti1Image(
                    data_plain,
                    cast(Any, vol_img).affine,
                    cast(Any, vol_img).header,
                )

    # Intensity scaling
    p_low, p_high = _validate_percentiles(
        float(args.auto_percentiles[0]), float(args.auto_percentiles[1])
    )
    if args.vmin is not None and args.vmax is not None:
        vmin = float(args.vmin)
        vmax = float(args.vmax)
    else:
        if args.intensity_mode == "global" and ndim == 4:
            auto_vmin, auto_vmax = _estimate_vmin_vmax_percentiles_4d_global(
                base_img,
                p_low=p_low,
                p_high=p_high,
                max_total_samples=int(args.auto_max_total_samples),
            )
        else:
            rng = np.random.default_rng(0)
            data = np.asanyarray(vol_img.dataobj)  # type: ignore
            samples = _sample_finite_values(
                data, max_samples=int(args.auto_max_samples), rng=rng
            )
            if samples.size > 0:
                auto_vmin = float(np.percentile(samples, p_low))
                auto_vmax = float(np.percentile(samples, p_high))
            else:
                auto_vmin = None
                auto_vmax = None

        vmin = float(args.vmin) if args.vmin is not None else auto_vmin
        vmax = float(args.vmax) if args.vmax is not None else auto_vmax

    # If we're plotting discrete atlas indices (0..N-1) as the base image and the user
    # didn't provide explicit bounds, use a deterministic normalization so colors match
    # legend entries exactly.
    if (
        atlas_only_mode
        and (not use_bg_as_base)
        and bool(args.atlas_legend)
        and atlas_discrete_vmin_vmax is not None
        and args.vmin is None
        and args.vmax is None
    ):
        vmin, vmax = atlas_discrete_vmin_vmax

    # Figure sizing
    width_px, height_px = int(args.size[0]), int(args.size[1])
    dpi = int(args.dpi)
    figsize = (width_px / dpi, height_px / dpi)

    n_views = len(views_cut_coords)
    ncols = int(args.ncols) if args.ncols is not None else n_views
    if ncols <= 0:
        raise ValueError(f"--ncols must be positive, got {ncols}")
    nrows = int(np.ceil(n_views / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()

    # Optional legend axis on the right.
    ax_legend = None
    if bool(args.atlas_legend) and len(atlas_legend_entries) > 0:
        fig.subplots_adjust(right=0.78)
        ax_legend = fig.add_axes((0.80, 0.08, 0.18, 0.84))
        ax_legend.set_axis_off()
        items = atlas_legend_entries
        if int(args.atlas_legend_max_items) > 0:
            items = items[: int(args.atlas_legend_max_items)]
        handles = [
            Patch(facecolor=rgba, edgecolor="none", label=f"{lid}: {name}")
            for (lid, name, rgba) in items
        ]
        ax_legend.legend(
            handles=handles,
            loc="center left",
            frameon=False,
            fontsize=float(args.atlas_legend_fontsize),
            handlelength=1.0,
            handleheight=1.0,
            labelspacing=0.35,
            borderaxespad=0.0,
        )

    displays = []
    for view_idx, cut_coords in enumerate(views_cut_coords):
        ax = axes_flat[view_idx]
        title = None
        if args.title:
            title = str(args.title).format(index=int(args.index), view=view_idx)

        # If the atlas is the base image and we're showing an atlas legend, the legend
        # is the intended key; suppress the colorbar to avoid redundancy.
        show_colorbar = bool(args.colorbar) and (view_idx == 0)
        if atlas_only_mode and bool(args.atlas_legend):
            show_colorbar = False

        if atlas_only_mode:
            # Atlas-only rendering uses plot_roi (built for discrete label images).
            # - Without --bg-img: no underlay (bg_img=None) and full opacity.
            # - With --bg-img: bg underlay is shown and atlas is alpha-blended.
            roi_img = (
                atlas_img_resampled if atlas_img_resampled is not None else vol_img
            )
            roi_cmap = (
                atlas_cmap_obj if atlas_cmap_obj is not None else str(args.atlas_cmap)
            )

            # In atlas-only mode, interpret --vmin/--vmax as ROI normalization.
            roi_vmin = float(args.vmin) if args.vmin is not None else None
            roi_vmax = float(args.vmax) if args.vmax is not None else None
            if (
                bool(args.atlas_legend)
                and atlas_discrete_vmin_vmax is not None
                and args.vmin is None
                and args.vmax is None
            ):
                roi_vmin, roi_vmax = atlas_discrete_vmin_vmax

            # plot_roi defaults to an MNI background; explicitly disable unless provided.
            roi_bg_img = vol_img if use_bg_as_base else None

            display = plotting.plot_roi(
                roi_img,
                bg_img=roi_bg_img,
                display_mode=args.display_mode,
                cut_coords=cut_coords,
                cmap=cast(Any, roi_cmap),
                alpha=(float(args.atlas_alpha) if use_bg_as_base else 1.0),
                # nilearn may replace NaNs with 0 internally; when 0 is background we
                # must threshold it out, otherwise it can take the first cmap color
                # and visually merge with the first label.
                threshold=(0.5 if bool(args.atlas_ignore_zero) else cast(Any, None)),
                black_bg=args.black_bg,
                colorbar=show_colorbar,
                annotate=bool(args.annotate),
                title=title,
                figure=fig,
                axes=ax,
                vmin=roi_vmin,
                vmax=roi_vmax,
                resampling_interpolation="nearest",
            )
            displays.append(display)
            continue

        # Non-atlas-only rendering: base image via plot_img + optional atlas overlay.
        bg_img_arg = str(args.bg_img) if args.bg_img is not None else None

        display = plotting.plot_img(
            vol_img,
            display_mode=args.display_mode,
            cut_coords=cut_coords,
            cmap=cast(Any, str(args.cmap)),
            vmin=vmin,
            vmax=vmax,
            black_bg=args.black_bg,
            bg_img=bg_img_arg,
            colorbar=show_colorbar,
            annotate=bool(args.annotate),
            title=title,
            figure=fig,
            axes=ax,
        )
        displays.append(display)

        if atlas_img_resampled is not None:
            overlay_kwargs: dict[str, Any] = {
                # nilearn's Display.add_overlay controls overlay opacity via
                # `transparency` (passed as `alpha` to matplotlib's imshow).
                "transparency": float(args.atlas_alpha),
                "cmap": cast(
                    Any,
                    (
                        atlas_cmap_obj
                        if atlas_cmap_obj is not None
                        else str(args.atlas_cmap)
                    ),
                ),
            }

            # For discrete label overlays (when legend is enabled), avoid nilearn's
            # default thresholding (would drop index 0) and avoid interpolation that
            # can create blended colors near parcel boundaries.
            if bool(args.atlas_legend) and atlas_cmap_obj is not None:
                overlay_kwargs.update(
                    {
                        # Threshold out background (0) if requested; otherwise avoid
                        # thresholding discrete labels.
                        "threshold": (0.5 if bool(args.atlas_ignore_zero) else None),
                        "interpolation": "nearest",
                        "resampling_interpolation": "nearest",
                    }
                )
                if (
                    atlas_discrete_vmin_vmax is not None
                    and args.vmin is None
                    and args.vmax is None
                ):
                    overlay_kwargs.update(
                        {
                            "vmin": float(atlas_discrete_vmin_vmax[0]),
                            "vmax": float(atlas_discrete_vmin_vmax[1]),
                        }
                    )

            cast(Any, display).add_overlay(atlas_img_resampled, **overlay_kwargs)

    for ax in axes_flat[n_views:]:
        ax.axis("off")

    if ax_legend is None:
        fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi)

    for d in displays:
        try:
            d.close()  # type: ignore
        except Exception:
            pass
    plt.close(fig)

    print(f"Wrote snapshot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
