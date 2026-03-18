"""Run a real-data lambda sweep for TemporalDecayEstimator.

Examples
--------
python scripts/tune_temporal_decay_estimator.py \
  --echo-files echo1.nii.gz echo2.nii.gz echo3.nii.gz echo4.nii.gz echo5.nii.gz \
  --mask brain_mask.nii.gz \
  --echo-times-ms 10.6,28.69,46.78,64.87,82.96 \
  --lambda0-values 0,0.1,1,3,10 \
  --lambda1-values 0,1,3,10,30,100 \
  --max-voxels 2000 \
  --results-csv results/temporal_decay_lambda_sweep.csv \
  --plot results/temporal_decay_lambda_sweep.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import nibabel as nib
import numpy as np
from nilearn.masking import apply_mask
from tedana.utils import make_adaptive_mask

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task_arousal.preprocess.components.multiecho_fit import TemporalDecayEstimator


@dataclass(frozen=True)
class TuningSummary:
    lambda0: float
    lambda1: float
    signal_nrmse: float
    log_s0_roughness: float
    beta1_roughness: float
    nan_fraction: float
    clip_fraction: float
    median_t2star_ms: float
    p99_t2star_ms: float


def _parse_float_list(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of floats.")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if any(number < 0 for number in values):
        raise argparse.ArgumentTypeError("Values must be non-negative.")
    return values


def _build_penalty_grid(
    lambda0_values: list[float], lambda1_values: list[float]
) -> list[tuple[float, float]]:
    return [(lambda0, lambda1) for lambda0 in lambda0_values for lambda1 in lambda1_values]


def _load_masked_echo_data(
    echo_files: list[str], mask_path: str
) -> tuple[np.ndarray, nib.nifti1.Nifti1Image]:
    mask_img = nib.nifti1.load(mask_path)
    if not isinstance(mask_img, nib.nifti1.Nifti1Image):
        raise TypeError(f"Expected NIfTI mask at {mask_path}, got {type(mask_img)}")

    masked_echoes: list[np.ndarray] = []
    for echo_file in echo_files:
        echo_img = nib.nifti1.load(echo_file)
        echo_masked = apply_mask(echo_img, mask_img).T
        masked_echoes.append(echo_masked)

    return np.stack(masked_echoes, axis=1), mask_img


def _sample_voxels(
    data: np.ndarray,
    masksum: np.ndarray,
    *,
    min_mask_sum: int,
    max_voxels: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible_idx = np.where(masksum >= min_mask_sum)[0]
    if eligible_idx.size == 0:
        raise ValueError(
            "No voxels satisfy the requested minimum adaptive-mask count. "
            "Lower --min-mask-sum or check the mask inputs."
        )

    if max_voxels is not None and eligible_idx.size > max_voxels:
        sampled_idx = np.sort(rng.choice(eligible_idx, size=max_voxels, replace=False))
    else:
        sampled_idx = eligible_idx

    return data[sampled_idx], masksum[sampled_idx], sampled_idx


def _second_difference_rms(curves: np.ndarray) -> np.ndarray:
    if curves.shape[1] < 3:
        return np.zeros(curves.shape[0], dtype=float)
    second_diff = curves[:, 2:] - 2.0 * curves[:, 1:-1] + curves[:, :-2]
    finite = np.isfinite(second_diff)
    counts = finite.sum(axis=1)
    rms = np.full(curves.shape[0], np.nan, dtype=float)
    valid = counts > 0
    if np.any(valid):
        squared = np.where(finite, second_diff**2, 0.0)
        rms[valid] = np.sqrt(squared[valid].sum(axis=1) / counts[valid])
    return rms


def _compute_signal_nrmse(
    observed: np.ndarray,
    reconstructed: np.ndarray,
    masksum: np.ndarray,
) -> float:
    echo_indices = np.arange(observed.shape[1])[None, :, None]
    used_echoes = echo_indices < masksum[:, None, None]
    finite = np.isfinite(observed) & np.isfinite(reconstructed)
    valid = used_echoes & finite
    if not np.any(valid):
        return np.nan

    residual = reconstructed[valid] - observed[valid]
    scale = np.median(np.abs(observed[valid]))
    if scale <= 0 or not np.isfinite(scale):
        return np.nan
    return float(np.sqrt(np.mean(residual**2)) / scale)


def _summarize_fit(
    s0_hat: np.ndarray,
    t2_hat: np.ndarray,
    observed: np.ndarray,
    echo_times_ms: np.ndarray,
    masksum: np.ndarray,
    max_t2star_ms: float,
    min_signal: float,
    lambda0: float,
    lambda1: float,
) -> TuningSummary:
    clipped_s0 = np.clip(s0_hat, a_min=min_signal, a_max=None)
    log_s0 = np.log(clipped_s0)
    beta1 = np.divide(
        1.0,
        t2_hat,
        out=np.full_like(t2_hat, np.nan, dtype=float),
        where=np.isfinite(t2_hat) & (t2_hat > 0),
    )

    reconstructed = clipped_s0[:, None, :] * np.exp(
        -echo_times_ms[None, :, None] / t2_hat[:, None, :]
    )
    signal_nrmse = _compute_signal_nrmse(observed, reconstructed, masksum)

    finite_t2 = np.isfinite(t2_hat)
    clipped = finite_t2 & np.isclose(t2_hat, max_t2star_ms)
    clip_fraction = float(clipped.sum() / finite_t2.sum()) if np.any(finite_t2) else np.nan

    log_s0_roughness = float(np.nanmedian(_second_difference_rms(log_s0)))
    beta1_roughness = float(np.nanmedian(_second_difference_rms(beta1)))
    nan_fraction = float(np.isnan(t2_hat).mean())
    median_t2star_ms = float(np.nanmedian(t2_hat))
    p99_t2star_ms = float(np.nanpercentile(t2_hat, 99))

    return TuningSummary(
        lambda0=lambda0,
        lambda1=lambda1,
        signal_nrmse=signal_nrmse,
        log_s0_roughness=log_s0_roughness,
        beta1_roughness=beta1_roughness,
        nan_fraction=nan_fraction,
        clip_fraction=clip_fraction,
        median_t2star_ms=median_t2star_ms,
        p99_t2star_ms=p99_t2star_ms,
    )


def _evaluate_grid(
    data: np.ndarray,
    masksum: np.ndarray,
    echo_times_ms: np.ndarray,
    penalties: list[tuple[float, float]],
    *,
    te_rescale_factor: float,
    min_signal: float,
    max_t2star_ms: float,
) -> list[TuningSummary]:
    summaries: list[TuningSummary] = []
    for lambda0, lambda1 in penalties:
        estimator = TemporalDecayEstimator(
            TE=echo_times_ms,
            T=data.shape[2],
            lambda0=lambda0,
            lambda1=lambda1,
            min_signal=min_signal,
            te_rescale_factor=te_rescale_factor,
            max_t2star_ms=max_t2star_ms,
        )
        s0_hat, t2_hat = estimator.fit(data=data, adaptive_mask=masksum)
        summaries.append(
            _summarize_fit(
                s0_hat=s0_hat,
                t2_hat=t2_hat,
                observed=data,
                echo_times_ms=echo_times_ms,
                masksum=masksum,
                max_t2star_ms=max_t2star_ms,
                min_signal=min_signal,
                lambda0=lambda0,
                lambda1=lambda1,
            )
        )
    return summaries


def _print_dataset_summary(
    masksum_all: np.ndarray,
    sampled_masksum: np.ndarray,
    sampled_idx: np.ndarray,
    penalties: list[tuple[float, float]],
) -> None:
    values, counts = np.unique(masksum_all, return_counts=True)
    distribution = ", ".join(
        f"{int(value)}:{int(count)}" for value, count in zip(values, counts, strict=True)
    )
    print(f"Adaptive-mask distribution (all voxels): {distribution}")
    print(f"Sampled voxels: {sampled_idx.size}")
    print(f"Sampled adaptive-mask median: {np.median(sampled_masksum):.1f}")
    print(f"Penalty pairs: {len(penalties)}")


def _print_summary_table(summaries: list[TuningSummary]) -> None:
    ordered = sorted(
        summaries,
        key=lambda summary: (
            np.inf if np.isnan(summary.signal_nrmse) else summary.signal_nrmse,
            np.inf if np.isnan(summary.clip_fraction) else summary.clip_fraction,
            np.inf if np.isnan(summary.nan_fraction) else summary.nan_fraction,
        ),
    )
    print(
        "lambda0\tlambda1\tsignal_nrmse\tlog_s0_roughness\tbeta1_roughness\t"
        "nan_fraction\tclip_fraction\tmedian_t2star_ms\tp99_t2star_ms"
    )
    for summary in ordered:
        print(
            f"{summary.lambda0:g}\t{summary.lambda1:g}\t{summary.signal_nrmse:.6f}\t"
            f"{summary.log_s0_roughness:.6f}\t{summary.beta1_roughness:.6f}\t"
            f"{summary.nan_fraction:.6f}\t{summary.clip_fraction:.6f}\t"
            f"{summary.median_t2star_ms:.3f}\t{summary.p99_t2star_ms:.3f}"
        )


def _write_csv(summaries: list[TuningSummary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=[
                "lambda0",
                "lambda1",
                "signal_nrmse",
                "log_s0_roughness",
                "beta1_roughness",
                "nan_fraction",
                "clip_fraction",
                "median_t2star_ms",
                "p99_t2star_ms",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)


def _plot_heatmaps(
    summaries: list[TuningSummary],
    lambda0_values: list[float],
    lambda1_values: list[float],
) -> Figure:
    summary_by_pair = {
        (summary.lambda0, summary.lambda1): summary for summary in summaries
    }

    metric_specs = [
        ("signal_nrmse", "Signal NRMSE"),
        ("log_s0_roughness", "log(S0) Roughness"),
        ("beta1_roughness", "beta1 Roughness"),
        ("nan_fraction", "NaN Fraction"),
        ("clip_fraction", "Clip Fraction"),
        ("p99_t2star_ms", "T2* P99 (ms)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (metric_name, title) in zip(axes.ravel(), metric_specs, strict=True):
        grid = np.full((len(lambda0_values), len(lambda1_values)), np.nan, dtype=float)
        for row, lambda0 in enumerate(lambda0_values):
            for col, lambda1 in enumerate(lambda1_values):
                summary = summary_by_pair[(lambda0, lambda1)]
                grid[row, col] = getattr(summary, metric_name)

        image = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("lambda1")
        ax.set_ylabel("lambda0")
        ax.set_xticks(np.arange(len(lambda1_values)))
        ax.set_xticklabels([f"{value:g}" for value in lambda1_values], rotation=30)
        ax.set_yticks(np.arange(len(lambda0_values)))
        ax.set_yticklabels([f"{value:g}" for value in lambda0_values])
        fig.colorbar(image, ax=ax, shrink=0.85)

    fig.suptitle("TemporalDecayEstimator lambda sweep", fontsize=16)
    return fig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--echo-files",
        nargs="+",
        required=True,
        help="Multi-echo NIfTI files ordered by echo time.",
    )
    parser.add_argument(
        "--mask",
        required=True,
        help="Brain mask NIfTI file used to sample voxels.",
    )
    parser.add_argument(
        "--echo-times-ms",
        type=_parse_float_list,
        required=True,
        help="Comma-separated echo times in milliseconds.",
    )
    parser.add_argument(
        "--lambda0-values",
        type=_parse_float_list,
        default=[0.0, 0.1, 1.0, 3.0, 10.0],
        help="Comma-separated lambda0 values for the sweep.",
    )
    parser.add_argument(
        "--lambda1-values",
        type=_parse_float_list,
        default=[0.0, 1.0, 3.0, 10.0, 30.0, 100.0],
        help="Comma-separated lambda1 values for the sweep.",
    )
    parser.add_argument(
        "--adaptive-threshold",
        type=int,
        default=2,
        help="Minimum number of usable echoes required by the adaptive mask.",
    )
    parser.add_argument(
        "--min-mask-sum",
        type=int,
        default=3,
        help="Minimum adaptive-mask count for voxels included in tuning metrics.",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=2000,
        help="Maximum number of voxels to sample for the sweep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when subsampling voxels.",
    )
    parser.add_argument(
        "--te-rescale-factor",
        type=float,
        default=100.0,
        help="Internal TE rescaling factor passed to TemporalDecayEstimator.",
    )
    parser.add_argument(
        "--min-signal",
        type=float,
        default=1e-6,
        help="Lower floor applied before log-fitting.",
    )
    parser.add_argument(
        "--max-t2star-ms",
        type=float,
        default=500.0,
        help="Upper cap applied to returned T2* estimates.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Optional CSV path for the full metric table.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional path for a heatmap summary figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the heatmap figure interactively.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if len(args.echo_files) != len(args.echo_times_ms):
        raise ValueError("The number of echo files must match the number of echo times.")
    if args.adaptive_threshold < 2:
        raise ValueError("adaptive-threshold must be at least 2.")
    if args.min_mask_sum < args.adaptive_threshold:
        raise ValueError(
            "min-mask-sum must be greater than or equal to adaptive-threshold."
        )
    if args.max_voxels is not None and args.max_voxels < 1:
        raise ValueError("max-voxels must be positive.")

    data, _ = _load_masked_echo_data(args.echo_files, args.mask)
    _, masksum = make_adaptive_mask(
        data,
        threshold=args.adaptive_threshold,
        methods=["dropout"],
    )

    rng = np.random.default_rng(args.seed)
    sampled_data, sampled_masksum, sampled_idx = _sample_voxels(
        data,
        masksum,
        min_mask_sum=args.min_mask_sum,
        max_voxels=args.max_voxels,
        rng=rng,
    )

    penalties = _build_penalty_grid(args.lambda0_values, args.lambda1_values)
    _print_dataset_summary(masksum, sampled_masksum, sampled_idx, penalties)
    summaries = _evaluate_grid(
        sampled_data,
        sampled_masksum,
        np.asarray(args.echo_times_ms, dtype=float),
        penalties,
        te_rescale_factor=args.te_rescale_factor,
        min_signal=args.min_signal,
        max_t2star_ms=args.max_t2star_ms,
    )
    _print_summary_table(summaries)

    if args.results_csv is not None:
        _write_csv(summaries, args.results_csv)
        print(f"Saved CSV summary to {args.results_csv}")

    if args.plot is not None or args.show:
        fig = _plot_heatmaps(summaries, args.lambda0_values, args.lambda1_values)
        if args.plot is not None:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.plot, dpi=150)
            print(f"Saved heatmap figure to {args.plot}")
        if args.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()