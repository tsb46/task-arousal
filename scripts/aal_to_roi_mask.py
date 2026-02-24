"""Convert an AAL fsLR32k atlas (GIFTI label.gii) into boolean vertex masks.

This script builds boolean vertex masks (left/right hemispheres) from one or more
ROI labels selected from per-hemisphere GIFTI label atlases in fsLR32k space.

Atlas source (AAL in fsLR32k):
    https://github.com/DiedrichsenLab/fs_LR_32

Examples
--------
List ROI names in the atlas:
    python scripts/aal_to_roi_mask.py --list-rois

Create a union mask from multiple ROIs (writes outprefix.L/R.func.gii):
    python scripts/aal_to_roi_mask.py --roi Precentral --roi Postcentral \
        --out results/central_gyrus
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import nibabel as nib

try:
    from nibabel.gifti.gifti import GiftiDataArray, GiftiImage
except Exception:  # pragma: no cover
    GiftiDataArray = None  # type: ignore[assignment]
    GiftiImage = None  # type: ignore[assignment]


DEFAULT_ATLAS_LEFT = Path("atlas/AAL.32k.L.label.gii")
DEFAULT_ATLAS_RIGHT = Path("atlas/AAL.32k.R.label.gii")


def _load_label_gii(
    path: str | Path,
) -> tuple[Any, np.ndarray, list[tuple[int, str]]]:
    """Load a GIFTI label atlas.

    Returns
    -------
    (img, labels, table)
        labels is a 1D array of integer label codes, one per vertex.
        table is a list of (key, name) pairs.
    """

    if GiftiImage is None:
        raise RuntimeError("nibabel.gifti is not available")

    img_any = nib.load(str(path))  # type: ignore[assignment]
    if not isinstance(img_any, GiftiImage):
        raise TypeError(f"Expected GIFTI .label.gii, got {type(img_any)}")
    img = img_any

    if not img.darrays:
        raise ValueError(f"No data arrays found in {path}")

    labels = np.asarray(img.darrays[0].data).ravel()
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels in {path}, got shape {labels.shape}")

    # Extract label table: nibabel stores this as a GiftiLabelTable.
    table_items: list[tuple[int, str]] = []
    lt = getattr(img, "labeltable", None)
    if lt is None or getattr(lt, "labels", None) is None:
        raise ValueError(f"No label table found in {path}")

    for lab in lt.labels:  # type: ignore[union-attr]
        key = int(getattr(lab, "key"))
        name = str(getattr(lab, "label"))
        table_items.append((key, name))

    return img, labels, table_items


def find_label_keys(
    label_table: Sequence[tuple[int, str]],
    roi_names: Sequence[str],
    *,
    match_mode: str = "exact",
    ignore_case: bool = False,
) -> list[int]:
    """Return integer label keys whose names match the requested ROI(s).

    Parameters
    ----------
    label_table:
        Sequence of (key, name) pairs.
    roi_names:
        ROI label names to match.
    match_mode:
        'exact' (default) or 'contains'.
    ignore_case:
        If True, compare case-insensitively.
    """

    if match_mode not in {"exact", "contains"}:
        raise ValueError(
            f"match_mode must be 'exact' or 'contains', got {match_mode!r}"
        )

    requested = [str(x) for x in roi_names]
    if ignore_case:
        requested_cmp = [x.casefold() for x in requested]
    else:
        requested_cmp = requested

    matches: list[int] = []
    for key, name in label_table:
        name_cmp = name.casefold() if ignore_case else name
        for req, req_cmp in zip(requested, requested_cmp, strict=True):
            if match_mode == "exact" and name_cmp == req_cmp:
                matches.append(key)
            elif match_mode == "contains" and req_cmp in name_cmp:
                matches.append(key)

    matches = sorted(set(int(k) for k in matches))
    if matches:
        return matches

    available = sorted(name for (_k, name) in label_table)
    raise KeyError(
        "No ROI labels matched. "
        f"Requested={list(roi_names)!r}, match_mode={match_mode!r}, ignore_case={ignore_case}. "
        f"Available examples: {available[:30]}"
    )


def vertex_mask_from_labels(
    labels: np.ndarray, *, label_keys: Sequence[int]
) -> np.ndarray:
    keys = np.asarray([int(k) for k in label_keys], dtype=np.int64)
    if keys.size == 0:
        raise ValueError("label_keys must not be empty")
    vals_i = np.asarray(np.round(labels), dtype=np.int64)
    return np.isin(vals_i, keys)


def _save_mask_gifti(path: Path, mask: np.ndarray) -> None:
    if GiftiImage is None or GiftiDataArray is None:
        raise RuntimeError("nibabel.gifti is not available; cannot write GIFTI")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(mask, dtype=np.float32)
    # Use SHAPE intent for a per-vertex mask.
    darray = GiftiDataArray(data=data, intent="NIFTI_INTENT_SHAPE")
    img = GiftiImage(darrays=[darray])
    nib.save(img, str(path))  # type: ignore[arg-type]


def _normalize_out_prefix(out: str | Path) -> Path:
    """Treat --out as a prefix, even if a .gii filename is provided."""
    p = Path(str(out))
    suffixes = [s.lower() for s in p.suffixes]
    if suffixes[-2:] == [".func", ".gii"]:
        return p.with_suffix("").with_suffix("")
    if suffixes and suffixes[-1] == ".gii":
        return p.with_suffix("")
    return p


def _infer_default_out_prefix(atlas_left: Path, atlas_right: Path) -> Path:
    """Infer a reasonable output prefix from atlas filenames.

    If atlas paths end with '.L.label.gii' and '.R.label.gii', use the common
    prefix without hemisphere suffix.
    """

    left = Path(atlas_left)
    right = Path(atlas_right)
    left_name = left.name
    right_name = right.name

    if left_name.endswith(".L.label.gii") and right_name.endswith(".R.label.gii"):
        base_left = left_name[: -len(".L.label.gii")]
        base_right = right_name[: -len(".R.label.gii")]
        if base_left == base_right:
            return left.with_name(base_left)

    # Fallback: strip .gii and .label if present.
    return _normalize_out_prefix(left)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build fsLR32k vertex masks from per-hemisphere GIFTI label atlases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--atlas-left",
        default=str(DEFAULT_ATLAS_LEFT),
        help="Path to left hemisphere .label.gii atlas",
    )
    p.add_argument(
        "--atlas-right",
        default=str(DEFAULT_ATLAS_RIGHT),
        help="Path to right hemisphere .label.gii atlas",
    )
    p.add_argument(
        "--roi",
        action="append",
        default=[],
        help="ROI name to include (repeatable).",
    )
    p.add_argument(
        "--match",
        choices=["exact", "contains"],
        default="exact",
        help="How to match --roi against atlas label names.",
    )
    p.add_argument(
        "--ignore-case",
        action="store_true",
        help="Case-insensitive ROI name matching.",
    )
    p.add_argument(
        "--list-rois",
        action="store_true",
        help="Print atlas ROI labels and exit.",
    )
    p.add_argument(
        "--out",
        required=False,
        default=None,
        help=(
            "Output prefix/path (without hemisphere suffix). Will write "
            "<out>.L.func.gii and <out>.R.func.gii."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    atlas_left = Path(str(args.atlas_left))
    atlas_right = Path(str(args.atlas_right))
    if not atlas_left.exists():
        raise FileNotFoundError(f"Atlas (left) not found: {atlas_left}")
    if not atlas_right.exists():
        raise FileNotFoundError(f"Atlas (right) not found: {atlas_right}")

    _imgL, labels_L, table_L = _load_label_gii(atlas_left)
    _imgR, labels_R, table_R = _load_label_gii(atlas_right)

    if bool(args.list_rois):
        # Many atlases use the same label table on both hemispheres; show left,
        # but also warn if right differs.
        left_items = sorted(table_L, key=lambda x: x[0])
        right_items = sorted(table_R, key=lambda x: x[0])

        if left_items != right_items:
            print("# NOTE: left/right label tables differ; printing both")
            print("# LEFT")
            for key, name in left_items:
                print(f"{key}\t{name}")
            print("# RIGHT")
            for key, name in right_items:
                print(f"{key}\t{name}")
        else:
            for key, name in left_items:
                print(f"{key}\t{name}")
        return 0

    rois: list[str] = [str(x) for x in (args.roi or [])]
    if not rois:
        raise ValueError("At least one --roi is required (or use --list-rois)")

    keys_L = find_label_keys(
        table_L,
        rois,
        match_mode=str(args.match),
        ignore_case=bool(args.ignore_case),
    )
    keys_R = find_label_keys(
        table_R,
        rois,
        match_mode=str(args.match),
        ignore_case=bool(args.ignore_case),
    )

    mask_L = vertex_mask_from_labels(labels_L, label_keys=keys_L)
    mask_R = vertex_mask_from_labels(labels_R, label_keys=keys_R)

    out_prefix = (
        _normalize_out_prefix(str(args.out))
        if args.out is not None
        else _infer_default_out_prefix(atlas_left, atlas_right)
    )

    out_L = Path(str(out_prefix) + ".L.func.gii")
    out_R = Path(str(out_prefix) + ".R.func.gii")
    _save_mask_gifti(out_L, mask_L)
    _save_mask_gifti(out_R, mask_R)

    print(
        "Built masks: "
        f"nL={int(mask_L.sum())} / {int(mask_L.size)}, "
        f"nR={int(mask_R.sum())} / {int(mask_R.size)}. "
        f"KeysL={keys_L}, KeysR={keys_R}, ROIs={rois}. "
        f"Outputs: {out_L} {out_R}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
