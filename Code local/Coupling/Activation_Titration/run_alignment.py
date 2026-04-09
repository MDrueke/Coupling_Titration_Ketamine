"""pulse registration pipeline: align NIDQ stimulus channel to AP timebase, split
into brain-state blocks, assign amplitudes from WaveformSequence files.

output: stim_amplitudes.csv with columns  onset_time_s | amplitude_v | brain_state
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomllib

from align_datastreams import DataStreamAligner
from match_amplitudes import match_amplitudes, match_amplitudes_sequential
from recording import _canonical_state, resolve_session_paths

RECOMPUTE = False  # set True to redo everything even if stim_amplitudes.csv exists


def load_config(path="config.toml"):
    with open(path, "rb") as f:
        return tomllib.load(f)


SYNC_PARAMS = {
    "target_duration_ms": 500.0,
    "tolerance_ms": 5.0,
    "merge_gap_ms": 3.0,  # merge noise fragments within a single sync pulse
    "max_trim": 1,
}
SYNC_BIT_AP = 6  # bit #6 of last channel in AP
SYNC_BIT_NIDQ = 0  # bit 0 (line 0) in NIDQ

STIM_PARAMS = {
    "target_duration_ms": 28.0,
    "tolerance_ms": 5.0,
    "merge_gap_ms": 3.0,
}


def _detect_anesthetic(session_dir: Path) -> str:
    """read the top-level meta.txt and return the canonical anesthetic state name."""
    meta_path = Path(session_dir) / "meta.txt"
    if not meta_path.exists():
        return "ketamine"
    for line in meta_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        canonical = _canonical_state(parts[0])
        if canonical is not None and canonical != "awake":
            return canonical
    return "ketamine"


def run_alignment(session_dir: Path, config: dict):
    paths = resolve_session_paths(session_dir)
    recording_dir = paths["recording_dir"]
    nidq_file = paths["nidq_file"]
    wf_csv_awake = paths["waveform_csv_awake"]
    wf_csv_anesthesia = paths["waveform_csv_anesthesia"]
    if wf_csv_anesthesia is None:
        raise FileNotFoundError(
            f"No anesthesia WaveformSequence CSV found in {session_dir}"
        )
    output_dir = recording_dir / config["files"]["output_dir"]
    output_dir.mkdir(exist_ok=True)

    anesthetic = _detect_anesthetic(session_dir)

    stim_file = output_dir / "stim_amplitudes.csv"
    if stim_file.exists() and not RECOMPUTE:
        print(
            f"Skipping {session_dir.name} — stim_amplitudes.csv exists "
            "(set RECOMPUTE=True to override)"
        )
        return

    # --- 1. NIDQ → AP alignment ---
    ap_file = recording_dir / recording_dir.name.replace("_imec0", "_t0.imec0.ap.cbin")
    if not ap_file.exists():
        ap_file = recording_dir / recording_dir.name.replace(
            "_imec0", "_t0.imec0.ap.bin"
        )
    if not ap_file.exists():
        raise FileNotFoundError(f"AP file not found in {recording_dir}")

    print("=" * 70)
    print(f"PULSE REGISTRATION: {session_dir.name}  [{anesthetic}]")
    print("=" * 70)

    aligner = DataStreamAligner(
        reference_file=ap_file,
        reference_sync_channel=-1,
        reference_sync_bit=SYNC_BIT_AP,
        sync_params=SYNC_PARAMS,
        cache_dir=output_dir,
    )
    aligner.add_target_stream(
        target_file=nidq_file,
        target_sync_channel=-1,
        target_sync_bit=SYNC_BIT_NIDQ,
        stream_name="nidq",
    )
    aligned_stim = aligner.align_channel(
        stream_name="nidq",
        channel_number=7,
        pulse_params=STIM_PARAMS,
    )
    _TEAL = "\033[38;2;187;230;228m"
    _RESET = "\033[0m"

    print(f"{_TEAL}\n  Aligned {len(aligned_stim)} pulses{_RESET}")

    # --- 2. split at largest ITI gap (awake → anesthesia transition) ---
    split_at = int(np.argmax(np.diff(aligned_stim))) + 1
    awake_pulses = aligned_stim[:split_at]
    anesthesia_pulses = aligned_stim[split_at:]
    print(
        f"{_TEAL}  Block split: {len(awake_pulses)} awake | {len(anesthesia_pulses)} {anesthetic}{_RESET}"
    )

    min_amp = config.get("alignment", {}).get("min_amplitude_v", None)

    # --- 3. amplitude matching ---
    # ketamine recordings: use histogram offset estimator + NN matching (handles missed pulses)
    # other anesthetics: sequential 1-to-1 matching (all pulses registered)
    print("\n  [awake]")
    if anesthetic == "ketamine":
        awake_matched = match_amplitudes(
            awake_pulses,
            str(wf_csv_awake),
            "awake",
            min_amplitude_v=min_amp,
            diag_dir=output_dir,
        )
        print(f"  [{anesthetic}]")
        anesthesia_matched = match_amplitudes(
            anesthesia_pulses,
            str(wf_csv_anesthesia),
            anesthetic,
            min_amplitude_v=min_amp,
            diag_dir=output_dir,
        )
        plot_match_residuals(output_dir)
    else:
        awake_matched = match_amplitudes_sequential(
            awake_pulses, str(wf_csv_awake), "awake"
        )
        print(f"  [{anesthetic}]")
        anesthesia_matched = match_amplitudes_sequential(
            anesthesia_pulses, str(wf_csv_anesthesia), anesthetic
        )

    # --- 4. combine and save ---
    awake_matched["brain_state"] = "awake"
    anesthesia_matched["brain_state"] = anesthetic
    stim_df = pd.concat([awake_matched, anesthesia_matched], ignore_index=True)
    stim_df.to_csv(stim_file, index=False)

    print(f"\nPULSE REGISTRATION COMPLETE")
    print(
        f"  {len(awake_matched)} awake pulses | {len(anesthesia_matched)} {anesthetic} pulses"
    )
    print(
        f"  Amplitude range: {stim_df['amplitude_v'].min():.4f} – "
        f"{stim_df['amplitude_v'].max():.4f} V"
    )
    print(f"  Output: {stim_file}")


def plot_match_residuals(output_dir: Path):
    """two-panel residual diagnostic for amplitude matching.

    red flags: step jump by ~ITI (matching shifted by one position, all subsequent
    amplitude assignments wrong); bimodal histogram; growing divergence (clock drift).
    """
    nature_style = {
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.labelsize": 14,
        "axes.linewidth": 1,
        "axes.titlesize": 16,
        "figure.facecolor": "white",
        "figure.figsize": (12, 8),
        "font.family": "sans-serif",
        "font.size": 12,
        "xtick.direction": "in",
        "xtick.labelsize": 12,
        "xtick.major.size": 5,
        "ytick.direction": "in",
        "ytick.labelsize": 12,
        "ytick.major.size": 5,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
    }

    blocks = [("awake", "steelblue"), ("keta", "darkorange")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    with plt.rc_context(nature_style):
        for row, (block, color) in enumerate(blocks):
            csv_path = output_dir / f"stim_times_{block}.csv"
            if not csv_path.exists():
                print(f"  [warn] {csv_path.name} not found — skipping residual plot")
                continue
            df = pd.read_csv(csv_path)
            matched = df[df["status"] == "matched"].copy()  # excludes matched_extrap
            missed = df[df["status"] == "missed_wf"]
            spur = df[df["status"] == "spurious_ap"]

            matched["residual_s"] = matched["actual_ap_s"] - matched["expected_ap_s"]
            r_ms = matched["residual_s"].values * 1e3
            t = matched["actual_ap_s"].values

            slope, intercept = np.polyfit(t, r_ms, 1)
            trend = slope * t + intercept

            ax_ts = axes[row, 0]
            ax_his = axes[row, 1]

            ax_ts.scatter(t, r_ms, s=16, color=color, alpha=0.7, linewidths=0)
            ax_ts.plot(
                t,
                trend,
                color="black",
                linewidth=1.2,
                linestyle="--",
                label=f"drift {slope * 1e3:.3f} µs/s",
            )
            ax_ts.axhline(0, color="black", linewidth=0.8, linestyle=":")
            ax_ts.set_xlabel("AP onset time (s)")
            ax_ts.set_ylabel("residual (ms)")
            ax_ts.set_title(
                f"{block} — {len(matched)} matched, "
                f"{len(missed)} missed WF, {len(spur)} spurious det"
            )
            ax_ts.legend(fontsize=10)

            ax_his.hist(r_ms, bins=40, color=color, alpha=0.8, edgecolor="white")
            ax_his.axvline(0, color="black", linewidth=0.8, linestyle=":")
            ax_his.set_xlabel("residual (ms)")
            ax_his.set_ylabel("count")
            ax_his.set_title(
                f"mean={r_ms.mean():.2f} ms  std={r_ms.std():.2f} ms  "
                f"max|r|={np.abs(r_ms).max():.2f} ms"
            )

    fig.tight_layout()
    out_path = output_dir / "match_residuals.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\n  Residual plot saved: {out_path.name}")


def main():
    config = load_config()
    with open("sessions.toml", "rb") as f:
        sessions_cfg = tomllib.load(f)

    for d in sessions_cfg["sessions"]["dirs"]:
        run_alignment(Path(d), config)


if __name__ == "__main__":
    main()
