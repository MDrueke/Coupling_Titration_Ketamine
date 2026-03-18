"""
Activation titration analysis pipeline.

Processes each session for two brain states (awake, ketamine), produces per-session
and pooled activation curves and PSTHs, and tests for state differences.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import tomllib

from recording import Recording, resolve_session_paths


# =============================================================================
# PARAMETERS
# =============================================================================

PULSE_WINDOW = (5.0, 30.0)           # ms post-onset for response measurement
BASELINE_EXCLUSION = (-100.0, 500.0) # ms: exclude this window around each pulse from baseline
LAYER_FILTER = "cortex"              # None, "cortex", or a specific layer e.g. "5"
ZSCORE = False
MIN_SPIKES_PER_STATE = 500           # minimum spikes in each brain state to include a neuron

PSTH_PRE_WINDOW = 200.0              # ms before onset
PSTH_POST_WINDOW = 500.0             # ms after onset
PSTH_BIN_SIZE = 5.0                  # ms
PSTH_AMPLITUDE_RANGE = (4.0, 5.0)   # (min, max) V — pulses used for PSTH
STIM_DURATION_MS = 28.0

ALPHA = 0.05                         # FDR threshold for Wilcoxon + B-H correction

COLOR_AWAKE = "#D6604D"              # orange-red
COLOR_KETA  = "#4393C3"              # blue

DEBUG = False

NATURE_STYLE = {
    "axes.axisbelow": True,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.labelcolor": "black",
    "axes.labelsize": 14,
    "axes.linewidth": 1,
    "axes.titlecolor": "black",
    "axes.titlesize": 16,
    "figure.facecolor": "white",
    "figure.figsize": (10, 6),
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "xtick.color": "black",
    "xtick.direction": "in",
    "xtick.labelsize": 12,
    "xtick.major.size": 5,
    "xtick.major.width": 1,
    "xtick.minor.size": 3,
    "xtick.minor.width": 0.5,
    "ytick.color": "black",
    "ytick.direction": "in",
    "ytick.labelsize": 12,
    "ytick.major.size": 5,
    "ytick.major.width": 1,
    "ytick.minor.size": 3,
    "ytick.minor.width": 0.5,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StateData:
    pulse_onsets:    np.ndarray     # (n_pulses,) s
    amplitudes:      np.ndarray     # (n_pulses,) V
    responses_df:    pd.DataFrame   # columns: unique_id, amplitude, response
    amplitude_stats: pd.DataFrame   # columns: amplitude, mean, sem, n_neurons
    bin_centers:     np.ndarray     # (n_bins,) ms
    psth:            np.ndarray     # (n_bins,) mean across neurons
    psth_sem:        np.ndarray     # (n_bins,)
    neuron_psths:    np.ndarray     # (n_neurons, n_bins) — kept for pooling


@dataclass
class SessionResult:
    session_name: str
    output_dir:   Path
    unit_ids:     List[int]
    unique_ids:   List[str]
    awake:        StateData
    ketamine:     StateData


# =============================================================================
# HELPERS
# =============================================================================

def load_config(config_path: str = "config.toml") -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def create_output_dir(recording_dir: Path, config: dict) -> Path:
    output_dir = recording_dir / config["files"]["output_dir"]
    output_dir.mkdir(exist_ok=True)
    return output_dir



def filter_neurons(
    rec: Recording, layer: Optional[str], min_spikes: int
) -> Tuple[List[int], List[str], str]:
    """
    Filter neurons by layer and minimum spike count in each brain state.

    Returns (unit_ids, unique_ids, session_name_placeholder).
    unique_ids are filled in process_session with the session prefix.
    """
    if not rec.stateTimes:
        raise ValueError("rec.stateTimes is empty — check meta.txt for awake/ketamine entries")

    df = rec.clusterInfo.copy()

    # layer filter
    if layer is not None:
        if layer.lower() == "cortex":
            cortical = ["1", "2/3", "4", "5", "6"]
            df = df[df["layer"].isin(cortical)]
        else:
            df = df[df["layer"] == layer]
    n_after_layer = len(df)

    # spike count filter: ≥ min_spikes in every state window
    kept = []
    for uid in df["cluster_id"]:
        spikes = rec.unitSpikes[uid]
        ok = True
        for start_min, end_min in rec.stateTimes.values():
            start_s = start_min * 60
            if np.isinf(end_min):
                n = np.sum(spikes >= start_s)
            else:
                n = np.sum((spikes >= start_s) & (spikes < end_min * 60))
            if n < min_spikes:
                ok = False
                break
        if ok:
            kept.append(uid)

    unit_ids = kept
    print(
        f"\t...{len(unit_ids)}/{n_after_layer} neurons passed spike count filter "
        f"(≥{min_spikes} per state)"
    )
    return unit_ids


def calculate_baseline_stats(
    rec: Recording,
    unit_ids: List[int],
    pulse_onsets: np.ndarray,
    exclusion_window: Tuple[float, float],
    state_window: Tuple[float, float],
) -> Dict[int, dict]:
    """
    Baseline firing rate within the state's time window, excluding pulse periods.

    state_window : (start_min, end_min); end_min may be np.inf
    """
    start_s = state_window[0] * 60
    if np.isinf(state_window[1]):
        all_times = np.concatenate(list(rec.unitSpikes.values()))
        end_s = float(np.max(all_times))
    else:
        end_s = state_window[1] * 60

    excl_start = pulse_onsets + exclusion_window[0] / 1000
    excl_end   = pulse_onsets + exclusion_window[1] / 1000
    total_excluded = len(pulse_onsets) * (exclusion_window[1] - exclusion_window[0]) / 1000
    baseline_duration = max((end_s - start_s) - total_excluded, 1.0)

    stats = {}
    for uid in unit_ids:
        spikes = rec.unitSpikes[uid]
        state_spikes = spikes[(spikes >= start_s) & (spikes < end_s)]
        in_baseline = np.ones(len(state_spikes), dtype=bool)
        for s, e in zip(excl_start, excl_end):
            in_baseline &= ~((state_spikes >= s) & (state_spikes < e))
        rate = np.sum(in_baseline) / baseline_duration
        stats[uid] = {
            "mean": rate,
            "std": np.sqrt(rate / baseline_duration),
            "duration": baseline_duration,
        }
    return stats


def calculate_responses(
    rec: Recording,
    unit_ids: List[int],
    unique_ids: List[str],
    pulse_onsets: np.ndarray,
    amplitudes: np.ndarray,
    pulse_window: Tuple[float, float],
    baseline_stats: Optional[Dict],
    zscore: bool,
) -> pd.DataFrame:
    """Firing rate (or z-score) per neuron per pulse."""
    win_start = pulse_window[0] / 1000
    win_end   = pulse_window[1] / 1000
    win_dur   = win_end - win_start

    rows = []
    for uid, uniq_id in zip(unit_ids, unique_ids):
        spikes = rec.unitSpikes[uid]
        if zscore and baseline_stats[uid]["std"] == 0:
            continue
        for onset, amp in zip(pulse_onsets, amplitudes):
            count = np.sum((spikes >= onset + win_start) & (spikes < onset + win_end))
            rate  = count / win_dur
            if zscore:
                b = baseline_stats[uid]
                resp = (rate - b["mean"]) / b["std"]
            else:
                resp = rate
            rows.append({"unique_id": uniq_id, "amplitude": amp, "response": resp})
    return pd.DataFrame(rows)


def aggregate_by_amplitude(responses_df: pd.DataFrame) -> pd.DataFrame:
    """Average per neuron per amplitude, then compute mean ± SEM across neurons."""
    per_neuron = (
        responses_df.groupby(["amplitude", "unique_id"])["response"]
        .mean()
        .reset_index()
    )
    result = (
        per_neuron.groupby("amplitude")["response"]
        .agg(
            mean="mean",
            sem=lambda x: x.std() / np.sqrt(len(x)),
            n_neurons="count",
        )
        .reset_index()
    )
    return result


def calculate_psth(
    rec: Recording,
    unit_ids: List[int],
    pulse_onsets: np.ndarray,
    amplitudes: np.ndarray,
    baseline_stats: Optional[Dict],
    zscore: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (bin_centers_ms, psth_mean, psth_sem, neuron_psths).
    neuron_psths has shape (n_neurons, n_bins) and is kept for pooling.
    """
    pre_s  = PSTH_PRE_WINDOW / 1000
    post_s = PSTH_POST_WINDOW / 1000
    bin_s  = PSTH_BIN_SIZE / 1000

    bin_edges  = np.arange(-pre_s, post_s + bin_s, bin_s)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 * 1000  # ms

    amp_mask = (amplitudes >= PSTH_AMPLITUDE_RANGE[0]) & (amplitudes <= PSTH_AMPLITUDE_RANGE[1])
    selected_onsets = pulse_onsets[amp_mask]
    if len(selected_onsets) == 0:
        raise ValueError(f"No pulses in amplitude range {PSTH_AMPLITUDE_RANGE}")

    neuron_psths = []
    for uid in unit_ids:
        spikes = rec.unitSpikes[uid]
        if zscore and baseline_stats[uid]["std"] == 0:
            continue
        trial_rates = []
        for onset in selected_onsets:
            rel = spikes - onset
            counts, _ = np.histogram(rel[(rel >= -pre_s) & (rel < post_s)], bins=bin_edges)
            trial_rates.append(counts / bin_s)
        neuron_avg = np.mean(trial_rates, axis=0)
        if zscore:
            b = baseline_stats[uid]
            neuron_avg = (neuron_avg - b["mean"]) / b["std"]
        neuron_psths.append(neuron_avg)

    if len(neuron_psths) == 0:
        raise ValueError("No valid neurons for PSTH")

    neuron_psths = np.array(neuron_psths)
    psth     = np.mean(neuron_psths, axis=0)
    psth_sem = np.std(neuron_psths, axis=0) / np.sqrt(len(neuron_psths))
    return bin_centers, psth, psth_sem, neuron_psths


def process_state(
    rec: Recording,
    unit_ids: List[int],
    unique_ids: List[str],
    pulse_onsets: np.ndarray,
    amplitudes: np.ndarray,
    state_window: Tuple[float, float],
) -> StateData:
    """Full analysis pipeline for one brain state."""
    baseline_stats = None
    if ZSCORE:
        baseline_stats = calculate_baseline_stats(
            rec, unit_ids, pulse_onsets, BASELINE_EXCLUSION, state_window
        )

    responses_df    = calculate_responses(rec, unit_ids, unique_ids, pulse_onsets, amplitudes,
                                          PULSE_WINDOW, baseline_stats, ZSCORE)
    amplitude_stats = aggregate_by_amplitude(responses_df)
    bin_centers, psth, psth_sem, neuron_psths = calculate_psth(
        rec, unit_ids, pulse_onsets, amplitudes, baseline_stats, ZSCORE
    )
    return StateData(
        pulse_onsets=pulse_onsets,
        amplitudes=amplitudes,
        responses_df=responses_df,
        amplitude_stats=amplitude_stats,
        bin_centers=bin_centers,
        psth=psth,
        psth_sem=psth_sem,
        neuron_psths=neuron_psths,
    )


# =============================================================================
# SESSION PROCESSING
# =============================================================================

def process_session(session_dir: Path, config: dict) -> SessionResult:
    """Full pipeline for one session. Returns SessionResult."""
    session_dir   = Path(session_dir)
    session_name  = session_dir.name
    paths         = resolve_session_paths(session_dir)
    recording_dir = paths["recording_dir"]
    output_dir    = create_output_dir(recording_dir, config)

    print("=" * 70)
    print(f"SESSION: {session_name}")
    print("=" * 70)

    # 1. Load pre-computed pulse table (produced by run_alignment.py)
    print("\n[1/3] Loading pulse table...")
    stim_file = output_dir / "stim_amplitudes.csv"
    if not stim_file.exists():
        raise FileNotFoundError(
            f"{stim_file} not found. Run run_alignment.py first."
        )
    stim_df      = pd.read_csv(stim_file)
    awake_df     = stim_df[stim_df["brain_state"] == "awake"]
    keta_df      = stim_df[stim_df["brain_state"] == "ketamine"]
    awake_onsets = awake_df["onset_time_s"].values
    awake_amps   = awake_df["amplitude_v"].values
    keta_onsets  = keta_df["onset_time_s"].values
    keta_amps    = keta_df["amplitude_v"].values
    print(f"  {len(awake_onsets)} awake pulses | {len(keta_onsets)} ketamine pulses")

    # 2. Load recording and filter neurons
    print("\n[2/3] Loading recording...")
    rec = Recording(recording_dir, config)
    unit_ids   = filter_neurons(rec, LAYER_FILTER, MIN_SPIKES_PER_STATE)
    unique_ids = [f"{session_name}_{uid}" for uid in unit_ids]
    print(f"  {len(unit_ids)} neurons included")

    # 3. Analyse each state
    print("\n[3/3] Computing responses...")
    print("  — Awake")
    awake_data = process_state(rec, unit_ids, unique_ids, awake_onsets, awake_amps,
                                rec.stateTimes["awake"])
    print("  — Ketamine")
    keta_data  = process_state(rec, unit_ids, unique_ids, keta_onsets, keta_amps,
                                rec.stateTimes["ketamine"])

    # save amplitude stats
    awake_data.amplitude_stats.to_csv(output_dir / "amplitude_response_awake.csv", index=False)
    keta_data.amplitude_stats.to_csv(output_dir  / "amplitude_response_keta.csv",  index=False)

    return SessionResult(
        session_name=session_name,
        output_dir=output_dir,
        unit_ids=unit_ids,
        unique_ids=unique_ids,
        awake=awake_data,
        ketamine=keta_data,
    )


# =============================================================================
# POOLING AND STATISTICS
# =============================================================================

def pool_sessions(results: List[SessionResult]) -> dict:
    """Concatenate per-neuron data across sessions and recompute group-level summaries."""
    awake_resp = pd.concat([r.awake.responses_df    for r in results], ignore_index=True)
    keta_resp  = pd.concat([r.ketamine.responses_df for r in results], ignore_index=True)

    awake_stats = aggregate_by_amplitude(awake_resp)
    keta_stats  = aggregate_by_amplitude(keta_resp)

    awake_psths = np.vstack([r.awake.neuron_psths    for r in results])
    keta_psths  = np.vstack([r.ketamine.neuron_psths for r in results])
    bin_centers = results[0].awake.bin_centers

    n = len(awake_psths)
    return {
        "awake_resp":    awake_resp,
        "keta_resp":     keta_resp,
        "awake_stats":   awake_stats,
        "keta_stats":    keta_stats,
        "bin_centers":   bin_centers,
        "awake_psth":    np.mean(awake_psths, axis=0),
        "awake_psth_sem": np.std(awake_psths, axis=0) / np.sqrt(n),
        "keta_psth":     np.mean(keta_psths,  axis=0),
        "keta_psth_sem": np.std(keta_psths,   axis=0) / np.sqrt(n),
        "n_neurons":     n,
        "n_sessions":    len(results),
    }


def _bh_correction(p_values: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns boolean reject array."""
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)
    order     = np.argsort(p_values)
    sorted_p  = np.asarray(p_values)[order]
    threshold = (np.arange(1, n + 1) / n) * alpha
    below     = sorted_p <= threshold
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    reject = np.zeros(n, dtype=bool)
    reject[order[:np.where(below)[0][-1] + 1]] = True
    return reject


def run_stats(awake_resp_df: pd.DataFrame, keta_resp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wilcoxon signed-rank test (per amplitude) on per-neuron mean responses.
    Multiple comparisons corrected with Benjamini-Hochberg.
    """
    awake_mean = (awake_resp_df.groupby(["unique_id", "amplitude"])["response"]
                  .mean().reset_index())
    keta_mean  = (keta_resp_df.groupby(["unique_id", "amplitude"])["response"]
                  .mean().reset_index())
    merged = awake_mean.merge(keta_mean, on=["unique_id", "amplitude"],
                               suffixes=("_awake", "_keta"))

    rows = []
    for amp in sorted(merged["amplitude"].unique()):
        sub = merged[merged["amplitude"] == amp]
        if len(sub) < 5:
            rows.append({"amplitude": amp, "statistic": np.nan, "p_value": 1.0, "n": len(sub)})
            continue
        stat, p = scipy.stats.wilcoxon(sub["response_awake"], sub["response_keta"])
        rows.append({"amplitude": amp, "statistic": stat, "p_value": p, "n": len(sub)})

    stats_df = pd.DataFrame(rows)
    stats_df["significant"] = _bh_correction(stats_df["p_value"].values)
    return stats_df


# =============================================================================
# PLOTTING
# =============================================================================

def plot_activation_curve(
    awake_stats: pd.DataFrame,
    keta_stats: pd.DataFrame,
    output_path: Path,
    stats_df: Optional[pd.DataFrame] = None,
    title: str = "Activation Titration Curve",
):
    """Overlay awake (orange) and ketamine (blue) activation curves."""
    plt.rcParams.update(NATURE_STYLE)
    fig, ax = plt.subplots()

    for stats, color, label in [
        (awake_stats, COLOR_AWAKE, "Awake"),
        (keta_stats,  COLOR_KETA,  "Ketamine"),
    ]:
        ax.errorbar(
            stats["amplitude"], stats["mean"], yerr=stats["sem"],
            color=color, marker="o", linestyle="-",
            capsize=4, capthick=1.5, label=label,
        )

    # significance markers at top of axes
    if stats_df is not None:
        sig_amps = stats_df.loc[stats_df["significant"], "amplitude"]
        for amp in sig_amps:
            ax.annotate("*", (amp, 1.04), xycoords=("data", "axes fraction"),
                        ha="center", fontsize=14, fontweight="bold")

    ax.set_xlabel("Stimulus Amplitude (V)")
    ax.set_ylabel("Z-scored Firing Rate" if ZSCORE else "Firing Rate (Hz)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_psth(
    awake_psth_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    keta_psth_data:  Tuple[np.ndarray, np.ndarray, np.ndarray],
    output_path: Path,
    title: str = "PSTH",
):
    """Overlay awake and ketamine PSTHs. No statistics."""
    plt.rcParams.update(NATURE_STYLE)
    fig, ax = plt.subplots(figsize=(6, 4))

    bin_centers, awake_psth, awake_sem = awake_psth_data
    _,           keta_psth,  keta_sem  = keta_psth_data

    ax.axvspan(0, STIM_DURATION_MS, color="lightgray", alpha=0.3, zorder=0)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    for psth, sem, color, label in [
        (awake_psth, awake_sem, COLOR_AWAKE, "Awake"),
        (keta_psth,  keta_sem,  COLOR_KETA,  "Ketamine"),
    ]:
        ax.plot(bin_centers, psth, color=color, linewidth=2, label=label)
        ax.fill_between(bin_centers, psth - sem, psth + sem,
                        color=color, alpha=0.2, linewidth=0)

    ax.set_xlabel("Time from Onset (ms)")
    ax.set_ylabel("Z-scored Firing Rate" if ZSCORE else "Firing Rate (Hz)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_session(result: SessionResult):
    """Save per-session activation curve and PSTH with both brain states overlaid."""
    plot_activation_curve(
        result.awake.amplitude_stats,
        result.ketamine.amplitude_stats,
        result.output_dir / "activation_curve.png",
        title=result.session_name,
    )
    plot_psth(
        (result.awake.bin_centers,    result.awake.psth,    result.awake.psth_sem),
        (result.ketamine.bin_centers, result.ketamine.psth, result.ketamine.psth_sem),
        result.output_dir / "psth.png",
        title=result.session_name,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = load_config()
    with open("sessions.toml", "rb") as f:
        sessions_cfg = tomllib.load(f)

    results = []
    for d in sessions_cfg["sessions"]["dirs"]:
        result = process_session(Path(d), config)
        plot_session(result)
        results.append(result)

    # Pooled analysis
    pooled_dir = Path("output_pooled")
    pooled_dir.mkdir(exist_ok=True)

    pooled   = pool_sessions(results)
    stats_df = run_stats(pooled["awake_resp"], pooled["keta_resp"])
    stats_df.to_csv(pooled_dir / "stats.csv", index=False)

    plot_activation_curve(
        pooled["awake_stats"], pooled["keta_stats"],
        pooled_dir / "activation_curve_pooled.png",
        stats_df=stats_df,
        title=f"Pooled ({pooled['n_neurons']} neurons, {pooled['n_sessions']} sessions)",
    )
    plot_psth(
        (pooled["bin_centers"], pooled["awake_psth"], pooled["awake_psth_sem"]),
        (pooled["bin_centers"], pooled["keta_psth"],  pooled["keta_psth_sem"]),
        pooled_dir / "psth_pooled.png",
        title=f"Pooled PSTH ({pooled['n_neurons']} neurons)",
    )

    sig_amps = stats_df.loc[stats_df["significant"], "amplitude"].tolist()
    print(f"\nPooled: {pooled['n_neurons']} neurons across {pooled['n_sessions']} sessions")
    print(f"Significant amplitudes (FDR α={ALPHA}): {sig_amps}")


if __name__ == "__main__":
    main()
