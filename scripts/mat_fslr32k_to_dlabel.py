"""scripts/mat_fslr32k_to_dlabel.py

Convert a MATLAB .mat atlas from the Network Connectivity Toolbox in fsLR 32k surface-vertex space into a CIFTI-2
label file (.dlabel.nii) using nibabel.

Assumptions
-----------
- The .mat contains *separate* left and right hemisphere label vectors.
- By default, we look for keys/fields named `lh_labels` and `rh_labels`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import nibabel as nib
from nibabel.loadsave import save as nib_save
from scipy.io import loadmat


N_VERT_32K = 32492


def _unwrap_singleton(x: Any) -> Any:
    """Unwrap common MATLAB->scipy containers like (1,1) object arrays."""
    if isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        return x.item()
    return x


def _has_field(obj: Any, key: str) -> bool:
    obj = _unwrap_singleton(obj)
    if isinstance(obj, dict):
        return key in obj
    if isinstance(obj, np.void) and obj.dtype.names is not None:
        return key in obj.dtype.names
    if hasattr(obj, "keys"):
        try:
            return key in obj.keys()  # type: ignore[operator]
        except Exception:
            pass
    if hasattr(obj, "__getitem__"):
        try:
            obj[key]
            return True
        except Exception:
            pass
    return hasattr(obj, key)


def _get_field(obj: Any, key: str) -> Any:
    obj = _unwrap_singleton(obj)
    if isinstance(obj, dict):
        return obj[key]
    if (
        isinstance(obj, np.void)
        and obj.dtype.names is not None
        and key in obj.dtype.names
    ):
        return obj[key]
    if hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        # dict-like objects
        try:
            return obj[key]
        except Exception:
            pass
    if hasattr(obj, key):
        return getattr(obj, key)
    if hasattr(obj, "__getitem__"):
        # last resort
        return obj[key]
    raise KeyError(key)


def _coerce_int_labels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.int32, copy=False)
    if np.all(np.isfinite(x)) and np.all(x == np.round(x)):
        return np.round(x).astype(np.int32)
    raise ValueError("Labels must be integers (or integer-valued floats).")


def _select_container(
    mat: dict, *, container: str | None, lh_key: str, rh_key: str
) -> tuple[Any, str]:
    """Return (container_object, container_name_for_logging)."""
    if container is not None:
        if container not in mat:
            available = sorted([k for k in mat.keys() if not k.startswith("__")])
            raise KeyError(
                f"--container {container!r} not found. Available top-level keys: {available}"
            )
        return mat[container], container

    # If the labels live at the top-level dict
    if lh_key in mat or rh_key in mat:
        return mat, "<toplevel>"

    # Otherwise: search non-private top-level entries for an object that has both fields.
    candidates: list[str] = [k for k in mat.keys() if not k.startswith("__")]
    for k in candidates:
        obj = mat[k]
        if _has_field(obj, lh_key) and _has_field(obj, rh_key):
            return obj, k

    raise ValueError(
        "Could not find a container holding both hemisphere label fields. "
        f"Looked for fields {lh_key!r} and {rh_key!r} in top-level keys: {sorted(candidates)}. "
        "Use --container to name the MATLAB struct/variable that holds the labels."
    )


def _load_hemi_labels(
    container_obj: Any, *, lh_key: str, rh_key: str
) -> tuple[np.ndarray, np.ndarray]:
    if not _has_field(container_obj, lh_key):
        raise KeyError(
            f"Missing left-hemisphere field {lh_key!r} in selected container"
        )
    if not _has_field(container_obj, rh_key):
        raise KeyError(
            f"Missing right-hemisphere field {rh_key!r} in selected container"
        )

    lh = _coerce_int_labels(np.asarray(_get_field(container_obj, lh_key)))
    rh = _coerce_int_labels(np.asarray(_get_field(container_obj, rh_key)))

    if lh.size != N_VERT_32K:
        raise ValueError(
            f"{lh_key!r} has length {lh.size}, expected {N_VERT_32K} for fsLR 32k"
        )
    if rh.size != N_VERT_32K:
        raise ValueError(
            f"{rh_key!r} has length {rh.size}, expected {N_VERT_32K} for fsLR 32k"
        )

    return lh, rh


def _rgba_for_label(k: int, alpha: float = 1.0) -> tuple[float, float, float, float]:
    # deterministic “good enough” colors without extra deps
    if k <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    r = ((k * 37) % 255) / 255.0
    g = ((k * 91) % 255) / 255.0
    b = ((k * 173) % 255) / 255.0
    return (float(r), float(g), float(b), float(alpha))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a fsLR 32k .mat atlas (vertex labels) to a CIFTI-2 .dlabel.nii using nibabel."
    )
    # Inputs/outputs can be passed positionally or via flags.
    # Keep positionals for convenience, but allow keyword-style CLI usage.
    parser.add_argument("mat_fp", nargs="?", type=Path, help="Input .mat (fsLR 32k)")
    parser.add_argument("out_fp", nargs="?", type=Path, help="Output .dlabel.nii")
    parser.add_argument(
        "--mat-fp",
        "--mat_fp",
        dest="mat_fp_opt",
        type=Path,
        default=None,
        help="Input .mat (fsLR 32k) (alternative to positional mat_fp)",
    )
    parser.add_argument(
        "--out-fp",
        "--out_fp",
        dest="out_fp_opt",
        type=Path,
        default=None,
        help="Output .dlabel.nii (alternative to positional out_fp)",
    )
    parser.add_argument(
        "--container",
        type=str,
        default=None,
        help=(
            "Top-level MATLAB variable name that contains the label fields. "
            "If omitted, will use top-level if possible or auto-detect a suitable container."
        ),
    )
    parser.add_argument(
        "--lh-key",
        type=str,
        default="lh_labels",
        help="Field/key name for LH labels inside the selected container (default: lh_labels)",
    )
    parser.add_argument(
        "--rh-key",
        type=str,
        default="rh_labels",
        help="Field/key name for RH labels inside the selected container (default: rh_labels)",
    )
    parser.add_argument(
        "--map-name", type=str, default="atlas", help="Map name inside the dlabel"
    )
    parser.add_argument(
        "--zero-is-unlabeled",
        action="store_true",
        help="Treat label 0 as unlabeled/transparent",
    )
    args = parser.parse_args()

    if args.mat_fp is not None and args.mat_fp_opt is not None:
        parser.error("Provide either positional mat_fp OR --mat-fp/--mat_fp, not both.")
    if args.out_fp is not None and args.out_fp_opt is not None:
        parser.error("Provide either positional out_fp OR --out-fp/--out_fp, not both.")

    mat_fp = args.mat_fp if args.mat_fp is not None else args.mat_fp_opt
    out_fp = args.out_fp if args.out_fp is not None else args.out_fp_opt
    if mat_fp is None or out_fp is None:
        parser.error(
            "Missing inputs. Use positional: mat_fp out_fp, or flags: --mat-fp ... --out-fp ..."
        )

    mat = loadmat(mat_fp, squeeze_me=True, struct_as_record=False)
    container_obj, container_name = _select_container(
        mat,
        container=args.container,
        lh_key=str(args.lh_key),
        rh_key=str(args.rh_key),
    )
    lh, rh = _load_hemi_labels(
        container_obj, lh_key=str(args.lh_key), rh_key=str(args.rh_key)
    )

    # Build fsLR32k BrainModelAxis (include *all* vertices in each hemi).
    from nibabel.cifti2.cifti2_axes import BrainModelAxis, LabelAxis

    verts = np.arange(N_VERT_32K, dtype=np.int32)
    bm_lh = BrainModelAxis.from_surface(verts, nvertex=N_VERT_32K, name="CORTEX_LEFT")
    bm_rh = BrainModelAxis.from_surface(verts, nvertex=N_VERT_32K, name="CORTEX_RIGHT")
    bm = bm_lh + bm_rh

    labels = np.concatenate([lh, rh]).astype(np.int32, copy=False)
    if labels.size != int(bm.size):
        raise RuntimeError("Internal size mismatch building BrainModelAxis.")

    unique = np.unique(labels)
    label_table: dict[int, tuple[str, tuple[float, float, float, float]]] = {}

    # Always include 0
    if args.zero_is_unlabeled:
        label_table[0] = ("Unlabeled", (0.0, 0.0, 0.0, 0.0))
    else:
        label_table[0] = ("Label0", _rgba_for_label(0, alpha=1.0))

    for k in unique:
        k_int = int(k)
        if k_int == 0:
            continue
        label_table[k_int] = (f"Network{k_int}", _rgba_for_label(k_int, alpha=1.0))

    label_axis = LabelAxis([args.map_name], [label_table])
    data = labels[np.newaxis, :]

    header = nib.cifti2.cifti2.Cifti2Header.from_axes((label_axis, bm))
    img = nib.cifti2.cifti2.Cifti2Image(data, header=header)
    nib_save(img, str(out_fp))

    print(f"Wrote: {out_fp}")
    print(f"Container: {container_name!r}")
    print(f"LH field: {str(args.lh_key)!r}")
    print(f"RH field: {str(args.rh_key)!r}")
    print(f"Unique labels: {np.unique(labels)}")


if __name__ == "__main__":
    main()
