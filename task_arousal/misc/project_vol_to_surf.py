"""task_arousal.misc.project_vol_to_surf

Utilities for projecting volumetric statistical maps onto cortical surfaces.

This module focuses on leveraging fMRIPrep derivatives that include FreeSurfer
reconstructions (fsnative surfaces) and ANTs transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import shutil
import subprocess
import tempfile

import numpy as np
import nibabel as nib

from nilearn import surface
from nilearn.image import resample_to_img
from nibabel.gifti.gifti import GiftiDataArray, GiftiImage

Hemi = Literal["L", "R"]


@dataclass(frozen=True)
class SurfaceProjection:
    """Result of projecting a volume to a single hemisphere surface."""

    hemi: Hemi
    texture: np.ndarray
    func_gii: GiftiImage
    out_func_gii: Path | None


def write_surface_projection_gifti(
    proj: SurfaceProjection,
    out_path: str | Path,
    *,
    overwrite: bool = False,
) -> SurfaceProjection:
    """Write a SurfaceProjection's GIFTI to disk.

    Parameters
    ----------
    proj : SurfaceProjection
        The projection to write.
    out_path : str | Path
        Output filepath for the `.func.gii`.
    overwrite : bool
        If False (default), raises if `out_path` exists.

    Returns
    -------
    SurfaceProjection
        A copy of `proj` with `out_func_gii` set to the resolved output path.
    """

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {out_path}")

    nib.save(proj.func_gii, str(out_path))  # type: ignore[attr-defined]
    return replace(proj, out_func_gii=out_path)


def write_surface_projections_gifti(
    projs: dict[Hemi, SurfaceProjection],
    out_dir: str | Path,
    *,
    out_basename: str,
    overwrite: bool = False,
) -> dict[Hemi, SurfaceProjection]:
    """Write multiple hemisphere projections to disk.

    Output filenames follow:
    `{out_basename}_hemi-{L|R}_space-fsnative.func.gii`
    """

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[Hemi, SurfaceProjection] = {}
    for hemi, proj in projs.items():
        out_path = out_dir / f"{out_basename}_hemi-{hemi}_space-fsnative.func.gii"
        written[hemi] = write_surface_projection_gifti(
            proj, out_path, overwrite=overwrite
        )
    return written


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


def _apply_t1w_mask_inplace(
    stat_t1w_fp: Path,
    mask_t1w_fp: Path,
    *,
    fill_value: float = 0.0,
) -> None:
    """Apply a 3D T1w-space brain mask to a 3D/4D T1w-space NIfTI in-place."""

    stat_img = nib.nifti1.load(str(stat_t1w_fp))
    if not isinstance(stat_img, nib.nifti1.Nifti1Image):
        raise TypeError(f"Expected NIfTI image for stat_t1w_fp, got {type(stat_img)}")

    mask_img = nib.nifti1.load(str(mask_t1w_fp))
    if not isinstance(mask_img, nib.nifti1.Nifti1Image):
        raise TypeError(f"Expected NIfTI image for mask_t1w_fp, got {type(mask_img)}")

    if mask_img.ndim != 3:
        raise ValueError(f"mask_t1w must be 3D, got shape {mask_img.shape}")

    # Resample mask to stat grid if needed.
    same_grid = mask_img.shape[:3] == stat_img.shape[:3]
    same_affine = (
        mask_img.affine is not None
        and stat_img.affine is not None
        and np.allclose(mask_img.affine, stat_img.affine)
    )
    if (not same_grid) or (not same_affine):
        mask_img = resample_to_img(
            source_img=mask_img,
            target_img=stat_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        assert isinstance(mask_img, nib.nifti1.Nifti1Image)

    mask = np.asanyarray(mask_img.dataobj) > 0.5
    stat = np.asanyarray(stat_img.dataobj)

    if stat.ndim == 3:
        stat_masked = np.where(mask, stat, fill_value)
    elif stat.ndim == 4:
        stat_masked = np.where(mask[..., None], stat, fill_value)
    else:
        raise ValueError(f"stat_t1w must be 3D or 4D, got shape {stat.shape}")

    out_img = nib.nifti1.Nifti1Image(
        np.asarray(stat_masked, dtype=np.float32),
        affine=stat_img.affine,
        header=stat_img.header.copy(),
    )
    out_img.update_header()
    nib.nifti1.save(out_img, str(stat_t1w_fp))


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
    mask_t1w: str | Path | Literal["fmriprep"] | None = "fmriprep",
    mask_fill_value: float = 0.0,
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
    mask_t1w : str | Path | 'fmriprep' | None
        Optional T1w-space mask to apply *after* MNI->T1w warping and *before* vol->surf.
        - None: no masking is applied.
        - 'fmriprep': auto-locate the subject's fMRIPrep T1w brain mask in the anat directory.
        - path: explicit path to a 3D mask NIfTI in T1w space.
    mask_fill_value : float
        Value assigned to voxels outside the mask (default 0.0).

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

    # Optional masking in T1w space to prevent sampling non-brain voxels.
    if mask_t1w is not None:
        if mask_t1w == "fmriprep":
            mask_fp = Path(f"{anat_dir}/sub-{subject}_desc-brain_mask.nii.gz")
        else:
            mask_fp = Path(mask_t1w)
        _require_exists(mask_fp, "T1w brain mask")
        _apply_t1w_mask_inplace(
            stat_t1w_fp=stat_t1w_fp,
            mask_t1w_fp=mask_fp,
            fill_value=float(mask_fill_value),
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
