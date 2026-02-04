"""task_arousal.misc.project_vol_to_surf

Utilities for projecting volumetric statistical maps onto cortical surfaces.

This module focuses on leveraging fMRIPrep derivatives that include FreeSurfer
reconstructions (fsnative surfaces) and ANTs transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import shutil
import subprocess
import tempfile

import numpy as np
import nibabel as nib

from nilearn import surface
from nibabel.gifti.gifti import GiftiDataArray, GiftiImage

Hemi = Literal["L", "R"]


@dataclass(frozen=True)
class SurfaceProjection:
    """Result of projecting a volume to a single hemisphere surface."""

    hemi: Hemi
    texture: np.ndarray
    func_gii: GiftiImage
    out_func_gii: Path | None


def _strip_nii_gz_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return Path(name).stem


def _require_exists(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")
    return path


def _find_single(anat_dir: Path, pattern: str, what: str) -> Path:
    matches = sorted(anat_dir.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find {what} in {anat_dir} (pattern={pattern!r})"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Found multiple candidates for {what} in {anat_dir} (pattern={pattern!r}): "
            + ", ".join(m.name for m in matches)
        )
    return matches[0]


def _run_checked(cmd: list[str]) -> None:
    """Run a command and raise a helpful error message on failure."""

    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        msg = f"Command failed: {' '.join(cmd)}"
        if stderr:
            msg += f"\nSTDERR:\n{stderr}"
        if stdout:
            msg += f"\nSTDOUT:\n{stdout}"
        raise RuntimeError(msg) from exc


def _gifti_from_texture(texture: np.ndarray) -> GiftiImage:
    """Create a functional GIFTI from a 1D (V,) or 2D (V, T) texture array."""

    texture = np.asarray(texture)
    if texture.ndim == 1:
        return GiftiImage(
            darrays=[GiftiDataArray(np.asarray(texture, dtype=np.float32))]
        )
    if texture.ndim == 2:
        darrays = [
            GiftiDataArray(np.asarray(texture[:, i], dtype=np.float32))
            for i in range(texture.shape[1])
        ]
        return GiftiImage(darrays=darrays)
    raise ValueError(f"Expected texture to be 1D or 2D, got shape {texture.shape}")


def _warp_mni_to_t1w(
    *,
    stat_mni: Path,
    t1w_ref: Path,
    xfm: Path,
    out_t1w: Path,
    ants_apply: str,
    ants_interp: str,
) -> None:
    """Warp a 3D or 4D MNI NIfTI to subject T1w using ANTs.

    For 4D inputs, this splits along time/volume and applies the transform to each
    3D volume, then merges back into a 4D NIfTI.
    """

    img = nib.nifti1.load(str(stat_mni))
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise TypeError(f"stat_mni must be a NIfTI image, got {type(img)}")

    ndim = img.ndim
    if ndim == 3:
        cmd = [
            ants_apply,
            "-d",
            "3",
            "-i",
            str(stat_mni),
            "-r",
            str(t1w_ref),
            "-o",
            str(out_t1w),
            "-t",
            str(xfm),
            "-n",
            ants_interp,
        ]
        _run_checked(cmd)
        return

    if ndim != 4:
        raise ValueError(f"stat_mni must be 3D or 4D, got shape {img.shape}")

    n_vols = img.shape[3]
    with tempfile.TemporaryDirectory(prefix="ants_split4d_") as tmp:
        tmp_dir = Path(tmp)
        out_vols: list[Path] = []
        affine = img.affine
        base_header = img.header.copy()
        for i in range(n_vols):
            vol_data = np.asanyarray(img.dataobj[..., i])
            vol_img = nib.nifti1.Nifti1Image(
                vol_data, affine=affine, header=base_header
            )
            in_fp = tmp_dir / f"in_{i:04d}.nii.gz"
            out_fp = tmp_dir / f"out_{i:04d}.nii.gz"
            nib.nifti1.save(vol_img, str(in_fp))
            cmd = [
                ants_apply,
                "-d",
                "3",
                "-i",
                str(in_fp),
                "-r",
                str(t1w_ref),
                "-o",
                str(out_fp),
                "-t",
                str(xfm),
                "-n",
                ants_interp,
            ]
            _run_checked(cmd)
            out_vols.append(out_fp)

        first = nib.nifti1.load(str(out_vols[0]))
        if not isinstance(first, nib.nifti1.Nifti1Image):
            raise RuntimeError("ANTs output was not a NIfTI image")
        out_affine = first.affine
        out_header = first.header.copy()
        stacked = np.stack(
            [np.asanyarray(nib.nifti1.load(str(p)).dataobj) for p in out_vols], axis=3
        ).astype(np.float32, copy=False)
        out_img = nib.nifti1.Nifti1Image(stacked, affine=out_affine, header=out_header)
        out_img.update_header()
        nib.nifti1.save(out_img, str(out_t1w))


def project_mni_stat_to_fsnative_surfaces(
    stat_mni: str | Path,
    *,
    subject: str,
    fmriprep_dir: str | Path,
    mni_space: str = "MNI152NLin2009cAsym",
    out_dir: str | Path | None = None,
    out_basename: str | None = None,
    hemis: tuple[Hemi, ...] = ("L", "R"),
    interpolation: Literal["linear", "nearest"] = "linear",
    n_samples: int = 10,
    keep_intermediate_t1w: bool = False,
) -> dict[Hemi, SurfaceProjection]:
    """Project an MNI-space statistical map to fsnative (subject) cortical surfaces.

    This function uses:
    1) fMRIPrep's ANTs transform to warp the statistic from MNI -> subject T1w.
    2) fMRIPrep's FreeSurfer-derived fsnative pial/white surfaces.
    3) `nilearn.surface.vol_to_surf` to sample the T1w-space volume onto the cortical ribbon.

    Requirements
    ------------
    - fMRIPrep run with FreeSurfer enabled.
    - `antsApplyTransforms` available on the system PATH.

    Parameters
    ----------
        stat_mni : str | Path
            Path to a 3D or 4D NIfTI statistical map in MNI space.
            For 4D inputs (e.g., FIR/HRF timecourses), each volume is warped and
            projected, and outputs are written as multi-map functional GIFTI.
    subject : str
            Subject label without the 'sub-' prefix.
    fmriprep_dir : str | Path
            Path to the fMRIPrep derivatives directory (the folder containing sub-*/).
    mni_space : str
            Name of the MNI template space corresponding to `stat_mni` and fMRIPrep transforms.
            Must match the fMRIPrep entity used in the transform filename.
    out_dir : str | Path | None
            If provided, writes projected `*.func.gii` files here.
    out_basename : str | None
            Base name used for output files (without extension). Defaults to the input stat filename.
    hemis : tuple['L'|'R', ...]
            Hemispheres to project.
    interpolation : {'linear', 'nearest'}
            Interpolation used for both MNI->T1w warping and vol->surf sampling.
    n_samples : int
            Number of samples between white and pial surfaces (higher is smoother, slower).
    keep_intermediate_t1w : bool
            If True, save the intermediate T1w-warped NIfTI to `out_dir`. Requires `out_dir`.

    Returns
    -------
    dict
            Mapping hemi -> SurfaceProjection.
    """

    stat_mni = Path(stat_mni)
    _require_exists(stat_mni, "MNI stat map")

    fmriprep_dir = Path(fmriprep_dir)
    anat_dir = fmriprep_dir / f"sub-{subject}" / "anat"
    _require_exists(anat_dir, "fMRIPrep anat directory")

    if keep_intermediate_t1w and out_dir is None:
        raise ValueError("keep_intermediate_t1w=True requires out_dir to be set")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    if out_basename is None:
        out_basename = _strip_nii_gz_suffix(stat_mni.name)

    # --- locate fMRIPrep anatomy and transform ---
    t1w_ref = anat_dir / f"sub-{subject}_desc-preproc_T1w.nii.gz"
    if not t1w_ref.exists():
        # Some fMRIPrep versions include additional entities in the filename.
        t1w_ref = _find_single(
            anat_dir, f"sub-{subject}_*desc-preproc_T1w.nii.gz", "T1w reference"
        )

    xfm = anat_dir / f"sub-{subject}_from-{mni_space}_to-T1w_mode-image_xfm.h5"
    if not xfm.exists():
        # Fall back to any MNI->T1w transform, but require it to be unique.
        xfm = _find_single(
            anat_dir,
            f"sub-{subject}_from-*_to-T1w_mode-image_xfm.h5",
            f"MNI({mni_space})->T1w transform",
        )

    ants_apply = shutil.which("antsApplyTransforms")
    if ants_apply is None:
        raise RuntimeError(
            "antsApplyTransforms was not found on PATH. "
            "Install ANTs (or load your neuroimaging module) to warp MNI->T1w using fMRIPrep .h5 transforms."
        )

    ants_interp = "Linear" if interpolation == "linear" else "NearestNeighbor"

    # Decide where the intermediate T1w-warped file lives.
    if out_dir is not None and keep_intermediate_t1w:
        stat_t1w_fp = out_dir / f"{out_basename}_space-T1w.nii.gz"
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="project_vol_to_surf_")
        stat_t1w_fp = Path(tmp_ctx.name) / "stat_space-T1w.nii.gz"

    # --- warp MNI -> T1w (3D or 4D) ---
    _warp_mni_to_t1w(
        stat_mni=stat_mni,
        t1w_ref=t1w_ref,
        xfm=xfm,
        out_t1w=stat_t1w_fp,
        ants_apply=ants_apply,
        ants_interp=ants_interp,
    )

    # --- project to surfaces ---
    results: dict[Hemi, SurfaceProjection] = {}
    for hemi in hemis:
        if hemi not in ("L", "R"):
            raise ValueError(f"Invalid hemi: {hemi!r}")

        pial = anat_dir / f"sub-{subject}_hemi-{hemi}_pial.surf.gii"
        white = anat_dir / f"sub-{subject}_hemi-{hemi}_white.surf.gii"
        if not pial.exists():
            pial = _find_single(
                anat_dir,
                f"sub-{subject}_hemi-{hemi}_*pial.surf.gii",
                f"{hemi} pial surface",
            )
        if not white.exists():
            white = _find_single(
                anat_dir,
                f"sub-{subject}_hemi-{hemi}_*white.surf.gii",
                f"{hemi} white surface",
            )

        texture = surface.vol_to_surf(
            img=str(stat_t1w_fp),
            surf_mesh=str(pial),
            inner_mesh=str(white),
            interpolation=interpolation,
            n_samples=n_samples,
        )
        texture = np.asarray(texture, dtype=np.float32)

        func_gii = _gifti_from_texture(texture)

        out_func = None
        if out_dir is not None:
            out_func = (
                Path(out_dir) / f"{out_basename}_hemi-{hemi}_space-fsnative.func.gii"
            )
            nib.save(func_gii, str(out_func))  # type: ignore[attr-defined]

        results[hemi] = SurfaceProjection(
            hemi=hemi, texture=texture, func_gii=func_gii, out_func_gii=out_func
        )

    # Ensure temp dir cleanup after we are done.
    if tmp_ctx is not None:
        tmp_ctx.cleanup()

    return results
