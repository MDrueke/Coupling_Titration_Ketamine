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
import scipy.ndimage
import scipy.stats
import tomllib

_TEAL = "\033[38;2;187;230;228m"
_RESET = "\033[0m"

from recording import Recording, resolve_session_paths

# =============================================================================
# PARAMETERS
# =============================================================================

PULSE_WINDOW = (5.0, 35.0)  # ms post-onset for response measurement
BASELINE_EXCLUSION = (
    -100.0,
    500.0,
)  # ms: exclude this window around each pulse from baseline
LAYER_FILTER = "cortex"  # None, "cortex", or a specific layer e.g. "5"
ZSCORE = False
MIN_SPIKES_PER_STATE = 100  # minimum spikes in each brain state to include a neuron

PSTH_PRE_WINDOW = 100.0  # ms before onset
PSTH_POST_WINDOW = 500.0  # ms after onset
PSTH_BIN_SIZE = 2.5  # ms
PSTH_AMPLITUDE_RANGE = (1.0, 5.0)  # (min, max) V — pulses used for PSTH
STIM_DURATION_MS = 28.0
INTERPOLATE_ARTIFACT = True  # interpolate across stimulation artifacts
ARTIFACT_WINDOWS_MS = [(0.0, 3.0), (27.0, 30.0)]  # (start, end) ms post-onset
RESPONSIVE_NEURON_DETECTION = "zeta"  # "zscore" or "zeta"
RESPONSIVE_ZSCORE_THRESHOLD = (
    4.0  # min mean z-score in PULSE_WINDOW (awake) to include neuron; None to disable
)
ZETA_MIN_AMPLITUDE = 2.0  # only trials with amplitude > this (V) are used for ZETA test
ZETA_MAX_DUR_MS = (
    35.0  # analysis window for ZETA test; excludes synaptic activation beyond this
)
PSTH_SMOOTH_SIGMA_MS = 2  # gaussian smoothing sigma in ms; 0 or None to disable
RASTER_N_NEURONS = 5  # top-N most responsive neurons shown in raster
RASTER_N_TRIALS = 20  # trials per neuron in raster (at highest stim amplitude)
RASTER_MIN_SPIKES = 0.5  # min mean spike count in stim period to be eligible for raster

ALPHA = 0.05  # FDR threshold for Wilcoxon + B-H correction

COLOR_AWAKE = "#D6604D"  # orange-red
COLOR_KETA = "#4393C3"  # blue

VOLTAGE_TO_mW = {
    "0.01": 0,
    "0.015": 0,
    "0.02": 0.1,
    "0.05": 0.6,
    "0.1": 1.5,
    "0.2": 3.1,
    "0.5": 7.8,
    "1": 14.7,
    "2": 26.7,
    "3": 37.1,
    "4": 45.8,
    "5": 54.7,
}

HEATMAP_CLIM_PERCENTILE = 98  # percentile used for heatmap color limits
HEATMAP_CMAP = "viridis"  # colormap for awake and ketamine panels
HEATMAP_DIFF_CMAP = "Greys"  # colormap for the difference panel
HEATMAP_FIGSIZE = (22, 10)  # fixed figure size for all heatmap plots
HEATMAP_FONT_SCALE = {
    "font.size": 24,
    "axes.labelsize": 28,
    "axes.titlesize": 32,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
}

DEBUG = False

NATURE_STYLE = {
    "axes.axisbelow": True,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.labelcolor": "black",
    "axes.labelsize": 14,
    "axes.linewidth": 2,
    "axes.titlecolor": "black",
    "axes.titlesize": 16,
    "figure.facecolor": "white",
    "figure.figsize": (6, 4),
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
    pulse_onsets: np.ndarray  # (n_pulses,) s
    amplitudes: np.ndarray  # (n_pulses,) V
    responses_df: pd.DataFrame  # columns: unique_id, amplitude, response
    amplitude_stats: pd.DataFrame  # columns: amplitude, mean, sem, n_neurons
    bin_centers: np.ndarray  # (n_bins,) ms
    psth: np.ndarray  # (n_bins,) mean across neurons
    psth_sem: np.ndarray  # (n_bins,)
    neuron_psths: np.ndarray  # (n_neurons, n_bins) — kept for pooling
    psth_unit_ids: List[int]  # unit_ids corresponding to neuron_psths rows
    raster_unit_ids: Optional[List[int]] = None  # top-N neurons for raster
    raster_spikes: Optional[List] = None  # [neuron][trial] = spike_times_ms


@dataclass
class SessionResult:
    session_name: str
    output_dir: Path
    unit_ids: List[int]  # responsive neurons (or all if threshold is None)
    unique_ids: List[str]
    awake: StateData  # responsive neurons only — for psth plot + activation curve
    ketamine: StateData
    awake_all: StateData  # all neurons passing layer/spike filter — for heatmap
    ketamine_all: StateData
    cluster_info: (
        pd.DataFrame
    )  # cluster_id, brain_depth, layer for ALL neurons (awake_all rows)


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
        raise ValueError(
            "rec.stateTimes is empty — check meta.txt for awake/ketamine entries"
        )

    df = rec.clusterInfo.copy()

    # layer filter
    if layer is not None:
        if layer.lower() == "cortex":
            cortical = ["1", "2/3", "4", "5", "6"]
            df = df[df["layer"].isin(cortical)]
        else:
            df = df[df["layer"] == layer]
    n_after_layer = len(df)
    if layer is not None:
        print(
            f"{_TEAL}\t...Analysis restricted to {layer}: {n_after_layer} neurons in target area{_RESET}"
        )
    layer_counts_pre = df["layer"].value_counts().sort_index()
    print(
        f"{_TEAL}\t...Layer counts before spike filter: { {k: int(v) for k, v in layer_counts_pre.items()} }{_RESET}"
    )

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
    layer_counts_post = (
        df[df["cluster_id"].isin(unit_ids)]["layer"].value_counts().sort_index()
    )
    print(
        f"{_TEAL}\t...Layer counts after spike filter:  { {k: int(v) for k, v in layer_counts_post.items()} }{_RESET}"
    )
    print(
        f"{_TEAL}\t...{len(unit_ids)}/{n_after_layer} neurons passed spike count filter "
        f"(≥{min_spikes} per state){_RESET}"
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
    excl_end = pulse_onsets + exclusion_window[1] / 1000
    total_excluded = (
        len(pulse_onsets) * (exclusion_window[1] - exclusion_window[0]) / 1000
    )
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


def _build_psth_bins() -> Tuple[np.ndarray, np.ndarray, List]:
    """Return (bin_edges_s, bin_centers_ms, artifact_idx_list) for the global PSTH parameters."""
    pre_s = PSTH_PRE_WINDOW / 1000
    post_s = PSTH_POST_WINDOW / 1000
    bin_s = PSTH_BIN_SIZE / 1000
    bin_edges = np.arange(-pre_s, post_s + bin_s, bin_s)
    bin_centers_ms = (bin_edges[:-1] + bin_edges[1:]) / 2 * 1000
    bin_edges_ms = bin_edges * 1000
    artifact_idx_list = []
    if INTERPOLATE_ARTIFACT:
        for art_start, art_end in ARTIFACT_WINDOWS_MS:
            mask = (bin_edges_ms[:-1] < art_end) & (bin_edges_ms[1:] > art_start)
            idx = np.where(mask)[0]
            if len(idx):
                artifact_idx_list.append(idx)
    return bin_edges, bin_centers_ms, artifact_idx_list


def _preprocess_trace(neuron_avg: np.ndarray, artifact_idx_list: List) -> np.ndarray:
    """Apply artifact interpolation and Gaussian smoothing to a trial-averaged trace."""
    for artifact_idx in artifact_idx_list:
        i0, i1 = artifact_idx[0] - 1, artifact_idx[-1] + 1
        if 0 <= i0 and i1 < len(neuron_avg):
            neuron_avg[artifact_idx] = np.interp(
                artifact_idx, [i0, i1], [neuron_avg[i0], neuron_avg[i1]]
            )
    if PSTH_SMOOTH_SIGMA_MS:
        neuron_avg = scipy.ndimage.gaussian_filter1d(
            neuron_avg, sigma=PSTH_SMOOTH_SIGMA_MS / PSTH_BIN_SIZE
        )
    return neuron_avg


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
    """
    Per-neuron per-amplitude response from preprocessed psth trace.
    Uses the same artifact interpolation and smoothing as calculate_psth.
    """
    bin_edges, bin_centers_ms, artifact_idx_list = _build_psth_bins()
    pre_s = PSTH_PRE_WINDOW / 1000
    post_s = PSTH_POST_WINDOW / 1000
    bin_s = PSTH_BIN_SIZE / 1000
    win_mask = (bin_centers_ms >= pulse_window[0]) & (bin_centers_ms <= pulse_window[1])

    # group onsets by amplitude once
    unique_amps = np.unique(amplitudes)
    amp_onsets = {amp: pulse_onsets[np.isclose(amplitudes, amp)] for amp in unique_amps}

    rows = []
    for uid, uniq_id in zip(unit_ids, unique_ids):
        if zscore and baseline_stats[uid]["std"] == 0:
            continue
        spikes = rec.unitSpikes[uid]
        for amp, onsets in amp_onsets.items():
            trial_rates = []
            for onset in onsets:
                rel = spikes - onset
                counts, _ = np.histogram(
                    rel[(rel >= -pre_s) & (rel < post_s)], bins=bin_edges
                )
                trial_rates.append(counts / bin_s)
            neuron_avg = _preprocess_trace(
                np.mean(trial_rates, axis=0), artifact_idx_list
            )
            if zscore:
                b = baseline_stats[uid]
                neuron_avg = (neuron_avg - b["mean"]) / b["std"]
            rows.append(
                {
                    "unique_id": uniq_id,
                    "amplitude": amp,
                    "response": float(np.mean(neuron_avg[win_mask])),
                }
            )
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
    Returns (bin_centers_ms, psth_mean, psth_sem, neuron_psths, psth_unit_ids).
    neuron_psths has shape (n_neurons, n_bins) and is kept for pooling.
    """
    bin_edges, bin_centers, artifact_idx_list = _build_psth_bins()
    pre_s = PSTH_PRE_WINDOW / 1000
    post_s = PSTH_POST_WINDOW / 1000
    bin_s = PSTH_BIN_SIZE / 1000

    amp_mask = (amplitudes >= PSTH_AMPLITUDE_RANGE[0]) & (
        amplitudes <= PSTH_AMPLITUDE_RANGE[1]
    )
    selected_onsets = pulse_onsets[amp_mask]
    if len(selected_onsets) == 0:
        raise ValueError(f"No pulses in amplitude range {PSTH_AMPLITUDE_RANGE}")

    neuron_psths = []
    psth_unit_ids = []
    for uid in unit_ids:
        spikes = rec.unitSpikes[uid]
        if zscore and baseline_stats[uid]["std"] == 0:
            continue
        trial_rates = []
        for onset in selected_onsets:
            rel = spikes - onset
            counts, _ = np.histogram(
                rel[(rel >= -pre_s) & (rel < post_s)], bins=bin_edges
            )
            trial_rates.append(counts / bin_s)
        neuron_avg = _preprocess_trace(np.mean(trial_rates, axis=0), artifact_idx_list)
        if zscore:
            b = baseline_stats[uid]
            neuron_avg = (neuron_avg - b["mean"]) / b["std"]
        neuron_psths.append(neuron_avg)
        psth_unit_ids.append(uid)

    if len(neuron_psths) == 0:
        raise ValueError("No valid neurons for PSTH")

    neuron_psths = np.array(neuron_psths)
    psth = np.mean(neuron_psths, axis=0)
    psth_sem = np.std(neuron_psths, axis=0) / np.sqrt(len(neuron_psths))
    return bin_centers, psth, psth_sem, neuron_psths, psth_unit_ids


def identify_responsive_neurons_zeta(
    rec,
    unit_ids: List[int],
    awake_onsets: np.ndarray,
    awake_amps: np.ndarray,
    keta_onsets: np.ndarray,
    keta_amps: np.ndarray,
) -> List[int]:
    """
    Run ZETA test per neuron using trials from both states with amplitude > ZETA_MIN_AMPLITUDE.
    Returns unit_ids with p <= ALPHA.
    """
    from zetapy import zetatest

    mask_awake = awake_amps > ZETA_MIN_AMPLITUDE
    mask_keta = keta_amps > ZETA_MIN_AMPLITUDE
    onsets = np.sort(np.concatenate([awake_onsets[mask_awake], keta_onsets[mask_keta]]))
    if len(onsets) == 0:
        print(
            f"  [ZETA] No trials with amplitude > {ZETA_MIN_AMPLITUDE} V — falling back to all trials"
        )
        onsets = np.sort(np.concatenate([awake_onsets, keta_onsets]))

    dur = ZETA_MAX_DUR_MS / 1000.0
    responsive = []
    for uid in unit_ids:
        p, *_ = zetatest(rec.unitSpikes[uid], onsets, dblUseMaxDur=dur)
        if p <= ALPHA:
            responsive.append(uid)
    return responsive


def identify_responsive_neurons(
    unit_ids: List[int],
    psth_unit_ids: List[int],
    neuron_psths: np.ndarray,
    baseline_stats: Dict,
    bin_centers: np.ndarray,
) -> List[int]:
    """
    Return unit_ids whose mean z-scored response in PULSE_WINDOW exceeds
    RESPONSIVE_ZSCORE_THRESHOLD (awake data, PSTH_AMPLITUDE_RANGE pulses).
    If ZSCORE=True, neuron_psths are already z-scored; otherwise baseline_stats
    are used to z-score inline for detection only.
    """
    win_mask = (bin_centers >= PULSE_WINDOW[0]) & (bin_centers <= PULSE_WINDOW[1])
    uid_to_row = {uid: i for i, uid in enumerate(psth_unit_ids)}
    responsive = []
    for uid in unit_ids:
        if uid not in uid_to_row:
            continue
        row = neuron_psths[uid_to_row[uid]]
        if not ZSCORE:
            b = baseline_stats[uid]
            if b["std"] == 0:
                continue
            row = (row - b["mean"]) / b["std"]
        if np.mean(row[win_mask]) > RESPONSIVE_ZSCORE_THRESHOLD:
            responsive.append(uid)
    return responsive


def _filter_state_to_responsive(
    state: StateData, unit_ids: List[int], unique_ids: List[str]
) -> StateData:
    """Return a copy of StateData restricted to the given unit_ids / unique_ids."""
    uid_set = set(unit_ids)
    uniq_set = set(unique_ids)

    # filter psth matrix
    row_mask = [i for i, uid in enumerate(state.psth_unit_ids) if uid in uid_set]
    new_psths = state.neuron_psths[row_mask]
    new_uids = [state.psth_unit_ids[i] for i in row_mask]
    new_psth = np.mean(new_psths, axis=0)
    new_sem = np.std(new_psths, axis=0) / np.sqrt(len(new_psths))

    # filter responses and recompute amplitude stats
    new_resp_df = state.responses_df[state.responses_df["unique_id"].isin(uniq_set)]
    new_amp_stats = aggregate_by_amplitude(new_resp_df)

    return StateData(
        pulse_onsets=state.pulse_onsets,
        amplitudes=state.amplitudes,
        responses_df=new_resp_df,
        amplitude_stats=new_amp_stats,
        bin_centers=state.bin_centers,
        psth=new_psth,
        psth_sem=new_sem,
        neuron_psths=new_psths,
        psth_unit_ids=new_uids,
    )


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

    responses_df = calculate_responses(
        rec,
        unit_ids,
        unique_ids,
        pulse_onsets,
        amplitudes,
        PULSE_WINDOW,
        baseline_stats,
        ZSCORE,
    )
    amplitude_stats = aggregate_by_amplitude(responses_df)
    bin_centers, psth, psth_sem, neuron_psths, psth_unit_ids = calculate_psth(
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
        psth_unit_ids=psth_unit_ids,
    )


# =============================================================================
# SESSION PROCESSING
# =============================================================================


def _collect_raster_spikes(
    rec, unit_ids: List[int], onsets: np.ndarray
) -> List[List[np.ndarray]]:
    """Per-trial spike times (ms re. onset) for each unit at the given onsets."""
    pre_s = PSTH_PRE_WINDOW / 1000
    post_s = PSTH_POST_WINDOW / 1000
    result = []
    for uid in unit_ids:
        spikes = rec.unitSpikes[uid]
        trials = []
        for onset in onsets[:RASTER_N_TRIALS]:
            mask = (spikes >= onset - pre_s) & (spikes < onset + post_s)
            trials.append((spikes[mask] - onset) * 1000)
        result.append(trials)
    return result


def process_session(session_dir: Path, config: dict) -> SessionResult:
    """Full pipeline for one session. Returns SessionResult."""
    session_dir = Path(session_dir)
    session_name = session_dir.name
    paths = resolve_session_paths(session_dir)
    recording_dir = paths["recording_dir"]
    output_dir = create_output_dir(recording_dir, config)

    print("=" * 70)
    print(f"SESSION: {session_name}")
    print("=" * 70)

    # 1. Load pre-computed pulse table (produced by run_alignment.py)
    print("\n[1/3] Loading pulse table...")
    stim_file = output_dir / "stim_amplitudes.csv"
    if not stim_file.exists():
        raise FileNotFoundError(f"{stim_file} not found. Run run_alignment.py first.")
    stim_df = pd.read_csv(stim_file)
    awake_df = stim_df[stim_df["brain_state"] == "awake"]
    keta_df = stim_df[stim_df["brain_state"] == "ketamine"]
    awake_onsets = awake_df["onset_time_s"].values
    awake_amps = awake_df["amplitude_v"].values
    keta_onsets = keta_df["onset_time_s"].values
    keta_amps = keta_df["amplitude_v"].values
    print(
        f"{_TEAL}  {len(awake_onsets)} awake pulses | {len(keta_onsets)} ketamine pulses{_RESET}"
    )

    # 2. Load recording and filter neurons
    print("\n[2/3] Loading recording...")
    rec = Recording(recording_dir, config)
    unit_ids = filter_neurons(rec, LAYER_FILTER, MIN_SPIKES_PER_STATE)
    unique_ids = [f"{session_name}_{uid}" for uid in unit_ids]
    print(f"{_TEAL}  {len(unit_ids)} neurons included{_RESET}")

    # 3. Analyse each state (all neurons passing layer + spike filter)
    print("\n[3/3] Computing responses...")
    print("  — Awake")
    awake_data_all = process_state(
        rec, unit_ids, unique_ids, awake_onsets, awake_amps, rec.stateTimes["awake"]
    )
    print("  — Ketamine")
    keta_data_all = process_state(
        rec, unit_ids, unique_ids, keta_onsets, keta_amps, rec.stateTimes["ketamine"]
    )

    # cluster_info for heatmap always covers all neurons (before responsiveness filter)
    cluster_info = (
        rec.clusterInfo[rec.clusterInfo["cluster_id"].isin(unit_ids)][
            ["cluster_id", "brain_depth", "layer"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    # restrict psth/activation-curve data to responsive neurons
    if RESPONSIVE_NEURON_DETECTION == "zeta":
        responsive_ids = identify_responsive_neurons_zeta(
            rec, unit_ids, awake_onsets, awake_amps, keta_onsets, keta_amps
        )
        responsive_unique = [f"{session_name}_{uid}" for uid in responsive_ids]
        print(
            f"{_TEAL}\t...{len(responsive_ids)}/{len(unit_ids)} responsive neurons "
            f"(ZETA p <= {ALPHA}, both states, amplitude > {ZETA_MIN_AMPLITUDE} V){_RESET}"
        )
        awake_data = _filter_state_to_responsive(
            awake_data_all, responsive_ids, responsive_unique
        )
        keta_data = _filter_state_to_responsive(
            keta_data_all, responsive_ids, responsive_unique
        )
        unit_ids = responsive_ids
        unique_ids = responsive_unique
    elif RESPONSIVE_ZSCORE_THRESHOLD is not None:
        awake_baseline_stats = (
            None
            if ZSCORE  # psths already z-scored
            else calculate_baseline_stats(
                rec, unit_ids, awake_onsets, BASELINE_EXCLUSION, rec.stateTimes["awake"]
            )
        )
        responsive_ids = identify_responsive_neurons(
            unit_ids,
            awake_data_all.psth_unit_ids,
            awake_data_all.neuron_psths,
            awake_baseline_stats,
            awake_data_all.bin_centers,
        )
        responsive_unique = [f"{session_name}_{uid}" for uid in responsive_ids]
        print(
            f"{_TEAL}\t...{len(responsive_ids)}/{len(unit_ids)} responsive neurons "
            f"(mean z > {RESPONSIVE_ZSCORE_THRESHOLD} in PULSE_WINDOW, awake){_RESET}"
        )
        awake_data = _filter_state_to_responsive(
            awake_data_all, responsive_ids, responsive_unique
        )
        keta_data = _filter_state_to_responsive(
            keta_data_all, responsive_ids, responsive_unique
        )
        unit_ids = responsive_ids
        unique_ids = responsive_unique
    else:
        awake_data = awake_data_all
        keta_data = keta_data_all

    # collect raster data for top-N neurons: z-score ranking + min spike count filter
    awake_max_onsets = awake_onsets[np.isclose(awake_amps, np.max(awake_amps))]
    keta_max_onsets = keta_onsets[np.isclose(keta_amps, np.max(keta_amps))]
    stim_end_s = STIM_DURATION_MS / 1000
    mean_stim_spikes = np.array(
        [
            np.mean(
                [
                    (
                        (rec.unitSpikes[uid] >= o)
                        & (rec.unitSpikes[uid] < o + stim_end_s)
                    ).sum()
                    for o in awake_max_onsets
                ]
            )
            for uid in awake_data.psth_unit_ids
        ]
    )
    win_mask = (awake_data.bin_centers >= PULSE_WINDOW[0]) & (
        awake_data.bin_centers <= PULSE_WINDOW[1]
    )
    responses = awake_data.neuron_psths[:, win_mask].mean(axis=1)
    eligible = np.where(mean_stim_spikes >= RASTER_MIN_SPIKES)[0]
    n_raster = min(RASTER_N_NEURONS, len(eligible))
    top_idx = eligible[np.argsort(responses[eligible])[::-1][:n_raster]]
    raster_uids = [awake_data.psth_unit_ids[i] for i in top_idx]
    awake_data.raster_unit_ids = raster_uids
    awake_data.raster_spikes = _collect_raster_spikes(
        rec, raster_uids, awake_max_onsets
    )
    keta_data.raster_unit_ids = raster_uids
    keta_data.raster_spikes = _collect_raster_spikes(rec, raster_uids, keta_max_onsets)

    # save amplitude stats (responsive neurons only)
    awake_data.amplitude_stats.to_csv(
        output_dir / "amplitude_response_awake.csv", index=False
    )
    keta_data.amplitude_stats.to_csv(
        output_dir / "amplitude_response_keta.csv", index=False
    )

    return SessionResult(
        session_name=session_name,
        output_dir=output_dir,
        unit_ids=unit_ids,
        unique_ids=unique_ids,
        awake=awake_data,
        ketamine=keta_data,
        awake_all=awake_data_all,
        ketamine_all=keta_data_all,
        cluster_info=cluster_info,
    )


# =============================================================================
# POOLING AND STATISTICS
# =============================================================================


def pool_sessions(results: List[SessionResult]) -> dict:
    """Concatenate per-neuron data across sessions and recompute group-level summaries."""
    awake_resp = pd.concat([r.awake.responses_df for r in results], ignore_index=True)
    keta_resp = pd.concat([r.ketamine.responses_df for r in results], ignore_index=True)

    awake_stats = aggregate_by_amplitude(awake_resp)
    keta_stats = aggregate_by_amplitude(keta_resp)

    awake_psths = np.vstack([r.awake.neuron_psths for r in results])
    keta_psths = np.vstack([r.ketamine.neuron_psths for r in results])
    bin_centers = results[0].awake.bin_centers

    n = len(awake_psths)

    # heatmap: use all neurons (awake_all/ketamine_all), common to both states per session
    hm_awake, hm_keta, ci_rows = [], [], []
    for r in results:
        a_idx = {uid: i for i, uid in enumerate(r.awake_all.psth_unit_ids)}
        k_idx = {uid: i for i, uid in enumerate(r.ketamine_all.psth_unit_ids)}
        common = [uid for uid in r.awake_all.psth_unit_ids if uid in k_idx]
        if not common:
            continue
        hm_awake.append(r.awake_all.neuron_psths[[a_idx[u] for u in common]])
        hm_keta.append(r.ketamine_all.neuron_psths[[k_idx[u] for u in common]])
        ci = (
            r.cluster_info.set_index("cluster_id")
            .loc[common][["brain_depth", "layer"]]
            .copy()
        )
        ci_rows.append(ci.reset_index())

    pooled_awake = np.vstack(hm_awake) if hm_awake else np.empty((0, len(bin_centers)))
    pooled_keta = np.vstack(hm_keta) if hm_keta else np.empty((0, len(bin_centers)))
    pooled_ci = pd.concat(ci_rows, ignore_index=True) if ci_rows else pd.DataFrame()

    # collect raster data from per-session top-N neurons for global re-ranking
    raster_psths, raster_awake_spikes, raster_keta_spikes = [], [], []
    for r in results:
        if not r.awake.raster_unit_ids:
            continue
        uid_to_row = {uid: i for i, uid in enumerate(r.awake.psth_unit_ids)}
        for i, uid in enumerate(r.awake.raster_unit_ids):
            if uid in uid_to_row:
                raster_psths.append(r.awake.neuron_psths[uid_to_row[uid]])
                raster_awake_spikes.append(r.awake.raster_spikes[i])
                raster_keta_spikes.append(r.ketamine.raster_spikes[i])
    pooled_raster_psths = (
        np.vstack(raster_psths) if raster_psths else np.empty((0, len(bin_centers)))
    )

    return {
        "awake_resp": awake_resp,
        "keta_resp": keta_resp,
        "awake_stats": awake_stats,
        "keta_stats": keta_stats,
        "bin_centers": bin_centers,
        "awake_psths": awake_psths,
        "keta_psths": keta_psths,
        "awake_psth": np.mean(awake_psths, axis=0),
        "awake_psth_sem": np.std(awake_psths, axis=0) / np.sqrt(n),
        "keta_psth": np.mean(keta_psths, axis=0),
        "keta_psth_sem": np.std(keta_psths, axis=0) / np.sqrt(n),
        "n_neurons": n,
        "n_sessions": len(results),
        "heatmap_awake": pooled_awake,
        "heatmap_keta": pooled_keta,
        "heatmap_cluster_info": pooled_ci,
        "raster_psths": pooled_raster_psths,
        "raster_awake_spikes": raster_awake_spikes,
        "raster_keta_spikes": raster_keta_spikes,
    }


def _bh_correction(p_values: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns boolean reject array."""
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p_values)
    sorted_p = np.asarray(p_values)[order]
    threshold = (np.arange(1, n + 1) / n) * alpha
    below = sorted_p <= threshold
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    reject = np.zeros(n, dtype=bool)
    reject[order[: np.where(below)[0][-1] + 1]] = True
    return reject


def run_stats(
    awake_resp_df: pd.DataFrame,
    keta_resp_df: pd.DataFrame,
    n_neurons: int,
    n_sessions: int,
) -> pd.DataFrame:
    """
    Wilcoxon signed-rank test (per amplitude) on per-neuron mean responses.
    Multiple comparisons corrected with Benjamini-Hochberg.
    """
    awake_mean = (
        awake_resp_df.groupby(["unique_id", "amplitude"])["response"]
        .mean()
        .reset_index()
    )
    keta_mean = (
        keta_resp_df.groupby(["unique_id", "amplitude"])["response"]
        .mean()
        .reset_index()
    )
    merged = awake_mean.merge(
        keta_mean, on=["unique_id", "amplitude"], suffixes=("_awake", "_keta")
    )

    rows = []
    for amp in sorted(merged["amplitude"].unique()):
        sub = merged[merged["amplitude"] == amp]
        if len(sub) < 5:
            rows.append(
                {"amplitude": amp, "statistic": np.nan, "p_value": 1.0, "n": len(sub)}
            )
            continue
        stat, p = scipy.stats.wilcoxon(sub["response_awake"], sub["response_keta"])
        rows.append({"amplitude": amp, "statistic": stat, "p_value": p, "n": len(sub)})

    stats_df = pd.DataFrame(rows)
    stats_df["significant"] = _bh_correction(stats_df["p_value"].values)
    stats_df["n_neurons"] = n_neurons
    stats_df["n_sessions"] = n_sessions
    return stats_df


def run_psth_stats(
    awake_psths: np.ndarray,
    keta_psths: np.ndarray,
    bin_centers: np.ndarray,
    n_neurons: int,
) -> pd.DataFrame:
    """Wilcoxon signed-rank test on per-neuron mean response in PULSE_WINDOW."""
    start, end = PULSE_WINDOW
    mask = (bin_centers >= start) & (bin_centers <= end)
    awake_vals = awake_psths[:, mask].mean(axis=1)
    keta_vals = keta_psths[:, mask].mean(axis=1)
    n = len(awake_vals)
    if n >= 5:
        stat, p = scipy.stats.wilcoxon(awake_vals, keta_vals)
    else:
        stat, p = np.nan, 1.0
    df = pd.DataFrame(
        [
            {
                "bin_start": start,
                "bin_end": end,
                "statistic": stat,
                "p_value": p,
                "n": n,
                "significant": bool(p <= ALPHA),
                "n_neurons": n_neurons,
            }
        ]
    )
    return df


# =============================================================================
# PLOTTING
# =============================================================================


_VOLT_TO_MW = {float(k): float(v) for k, v in VOLTAGE_TO_mW.items()}
# fit calibration curve on non-zero-output points (sub-threshold points excluded)
_cal_pairs = sorted((v, mw) for v, mw in _VOLT_TO_MW.items() if mw > 0)
_cal_v = np.array([v for v, mw in _cal_pairs])
_cal_mw = np.array([mw for v, mw in _cal_pairs])
_cal_poly = np.polyfit(_cal_v, _cal_mw, 3)


def _volt_to_mw(v: float) -> float:
    """Convert voltage (V) to mW/mm² using a degree-3 polynomial fit of the calibration data."""
    return float(max(0.0, np.polyval(_cal_poly, float(v))))


def _save(output_path: Path, **kwargs):
    """Save current figure as both .pdf and .png with tight bounding box."""
    kwargs.setdefault("bbox_inches", "tight")
    plt.savefig(output_path.with_suffix(".pdf"), transparent=True, **kwargs)
    plt.savefig(output_path.with_suffix(".png"), transparent=True, **kwargs)


def plot_calibration(output_path: Path):
    """Scatter of calibration points + polynomial fit trace."""
    plt.rcParams.update(NATURE_STYLE)
    fig, ax = plt.subplots(figsize=(6, 4))

    all_v = np.array(sorted(_VOLT_TO_MW))
    all_mw = np.array([_VOLT_TO_MW[v] for v in all_v])
    ax.scatter(all_v, all_mw, color="black", zorder=3, label="Calibration points")

    v_fit = np.linspace(all_v.min(), all_v.max(), 300)
    mw_fit = np.clip(np.polyval(_cal_poly, v_fit), 0, None)
    ax.plot(v_fit, mw_fit, color="steelblue", label="Degree-3 fit")

    ax.set_xlim(all_v.min(), all_v.max())
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Illumination Intensity (mW/mm²)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(output_path, dpi=300)
    plt.close()


def plot_activation_curve(
    awake_stats: pd.DataFrame,
    keta_stats: Optional[pd.DataFrame],
    output_path: Path,
    stats_df: Optional[pd.DataFrame] = None,
    title: str = "Activation Titration Curve",
    show_legend: bool = True,
):
    """Overlay awake (orange) and optionally ketamine (blue) activation curves."""
    plt.rcParams.update(NATURE_STYLE)
    fig, ax = plt.subplots()

    pairs = [(awake_stats, COLOR_AWAKE, "Awake")]
    if keta_stats is not None:
        pairs.append((keta_stats, COLOR_KETA, "Ketamine"))
    for stats, color, label in pairs:
        mw = stats["amplitude"].map(_volt_to_mw)
        ax.plot(mw, stats["mean"], color=color, marker="o", label=label)
        ax.fill_between(
            mw,
            stats["mean"] - stats["sem"],
            stats["mean"] + stats["sem"],
            color=color,
            alpha=0.25,
            linewidth=0,
        )

    # significance markers at top of axes
    if stats_df is not None:
        sig_amps = stats_df.loc[stats_df["significant"], "amplitude"]
        for amp in sig_amps:
            ax.annotate(
                "*",
                (_volt_to_mw(amp), 1.04),
                xycoords=("data", "axes fraction"),
                ha="center",
                fontsize=14,
                fontweight="bold",
            )

    all_mw = awake_stats["amplitude"].map(_volt_to_mw)
    if keta_stats is not None:
        all_mw = pd.concat([all_mw, keta_stats["amplitude"].map(_volt_to_mw)])
    ax.set_xlim(all_mw.min(), all_mw.max())
    ax.set_xticks([4, 25, 50])
    ax.set_xlabel("Illumination Intensity (mW/mm²)")
    ax.set_ylabel("Z-scored Firing Rate" if ZSCORE else "Firing Rate (Hz)")
    if show_legend:
        ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ZSCORE:
        all_means = [awake_stats["mean"].max()]
        if keta_stats is not None:
            all_means.append(keta_stats["mean"].max())
        top_mean = max(all_means)
        top_tick = int(np.ceil(top_mean / 100) * 100)
        yticks = [0, top_tick // 2, top_tick]
        ax.set_ylim(bottom=0)
    else:
        yticks = np.round(np.linspace(0, ax.get_ylim()[1], 4)).astype(int)
    ax.set_yticks(yticks)
    plt.tight_layout()
    _save(output_path, dpi=300)
    plt.close()


def plot_psth(
    awake_psth_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    keta_psth_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    output_path: Path,
    title: str = "PSTH",
    psth_stats_df: Optional[pd.DataFrame] = None,
    awake_raster: Optional[List] = None,
    keta_raster: Optional[List] = None,
):
    """Overlay awake and ketamine PSTHs, optionally with a raster panel above."""
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update(NATURE_STYLE)
    has_raster = awake_raster is not None and len(awake_raster) > 0

    if has_raster:
        fig = plt.figure(figsize=(6, 6))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0)
        ax_raster = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(6, 4))

    bin_centers, awake_psth, awake_sem = awake_psth_data
    _, keta_psth, keta_sem = keta_psth_data

    ax.axvspan(0, STIM_DURATION_MS, color="lightgray", alpha=0.3, zorder=0)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    for psth, sem, color, label in [
        (awake_psth, awake_sem, COLOR_AWAKE, "Awake"),
        (keta_psth, keta_sem, COLOR_KETA, "Ketamine"),
    ]:
        ax.plot(bin_centers, psth, color=color, linewidth=2, label=label)
        ax.fill_between(
            bin_centers, psth - sem, psth + sem, color=color, alpha=0.2, linewidth=0
        )

    max_y = max((awake_psth + awake_sem).max(), (keta_psth + keta_sem).max())
    bar_y = 1.05 * max_y

    if psth_stats_df is not None:
        sig_bins = psth_stats_df[psth_stats_df["significant"]]
        for _, row in sig_bins.iterrows():
            ax.hlines(
                bar_y, row["bin_start"], row["bin_end"], colors="black", linewidth=2
            )
        if len(sig_bins):
            ax.set_ylim(top=bar_y * 1.1)

    xmin, xmax = -PSTH_PRE_WINDOW, PSTH_POST_WINDOW
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Time from Onset (ms)")
    ax.set_ylabel("Z-scored Firing Rate" if ZSCORE else "Firing Rate (Hz)")
    if has_raster:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], color=COLOR_AWAKE, linewidth=2, label="Awake"),
            Line2D([0], [0], color=COLOR_KETA, linewidth=2, label="Ketamine"),
        ]
        ax_raster.legend(handles=handles, frameon=False, loc="upper right")
    else:
        ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ZSCORE:
        top_mean = max(awake_psth.max(), keta_psth.max())
        top_tick = int(np.ceil(top_mean / 100) * 100)
        yticks = [0, top_tick // 2, top_tick]
    else:
        yticks = np.round(np.linspace(*ax.get_ylim(), 4)).astype(int)
        yticks[np.argmin(np.abs(yticks))] = 0
    ax.set_yticks(yticks)

    if has_raster:
        n = RASTER_N_TRIALS
        neuron_gap = 1
        group_gap = 0
        N = len(awake_raster)
        keta_offset = N * (n + neuron_gap) + group_gap

        ax_raster.axvspan(0, STIM_DURATION_MS, color="lightgray", alpha=0.3, zorder=0)
        ax_raster.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        for neuron_i in range(N):
            awake_y_base = neuron_i * (n + neuron_gap)
            keta_y_base = keta_offset + neuron_i * (n + neuron_gap)
            for trial_j, spikes in enumerate(awake_raster[neuron_i]):
                y = awake_y_base + trial_j
                ax_raster.vlines(spikes, y, y + 0.85, color=COLOR_AWAKE, linewidth=4.0)
            for trial_j, spikes in enumerate(keta_raster[neuron_i]):
                y = keta_y_base + trial_j
                ax_raster.vlines(spikes, y, y + 0.85, color=COLOR_KETA, linewidth=4.0)

        total_y = keta_offset + N * (n + neuron_gap)
        ax_raster.set_ylim(-0.5, total_y)
        ax_raster.set_yticks([])
        ax_raster.spines[["right", "left", "top", "bottom"]].set_visible(False)
        ax_raster.set_xlim(xmin, xmax)
        ax_raster.set_xticks([])

    ax.set_xticks([0, int(xmax // 2), int(xmax)])
    ax.set_xticklabels(["0", str(int(xmax // 2)), str(int(xmax))])

    plt.tight_layout()
    _save(output_path, dpi=300)
    plt.close()


def _get_layer_boundaries(layers: List[str]):
    """Return (boundary_positions, [(layer_name, midpoint_y), ...]) from an ordered list."""
    boundaries = []
    label_info = []
    prev = layers[0]
    start = 0
    for i, lyr in enumerate(layers):
        if lyr != prev:
            boundaries.append(i - 0.5)
            label_info.append((prev, (start + i - 1) / 2))
            start = i
            prev = lyr
    label_info.append((prev, (start + len(layers) - 1) / 2))
    return boundaries, label_info


def _draw_psth_heatmap(
    awake_mat: np.ndarray,
    keta_mat: np.ndarray,
    diff_mat: np.ndarray,
    cluster_info: pd.DataFrame,
    bin_centers: np.ndarray,
    title: str,
    output_path: Path,
):
    """Shared renderer for three-panel PSTH heatmap: awake | ketamine | difference."""
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.rcParams.update(NATURE_STYLE)
    plt.rcParams.update(HEATMAP_FONT_SCALE)

    layers = cluster_info["layer"].fillna("?").tolist()
    boundaries, label_info = _get_layer_boundaries(layers)
    n_neurons = len(awake_mat)

    combined = np.concatenate([awake_mat.ravel(), keta_mat.ravel()])
    vmax = np.percentile(combined, HEATMAP_CLIM_PERCENTILE)
    vmin = np.percentile(combined, 100 - HEATMAP_CLIM_PERCENTILE) if ZSCORE else 0.0
    diff_lim = np.percentile(np.abs(diff_mat), HEATMAP_CLIM_PERCENTILE)

    ylabel = "Z-score" if ZSCORE else "Firing Rate (Hz)"
    diff_ylabel = "ΔZ" if ZSCORE else "ΔHz"

    # layout: awake | keta | colorbar | spacer | diff  (diff gets its own colorbar appended right)
    fig = plt.figure(figsize=HEATMAP_FIGSIZE)
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 0.06, 0.25, 1], wspace=0.1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax_main = fig.add_subplot(gs[2])
    ax2 = fig.add_subplot(gs[4])

    # extent: [xmin, xmax, ymax (bottom), ymin (top)] with origin="upper"
    extent = [bin_centers[0], bin_centers[-1], n_neurons - 0.5, -0.5]

    for ax, mat, panel_title, cmap, vm_min, vm_max in [
        (ax0, awake_mat, "Awake", HEATMAP_CMAP, vmin, vmax),
        (ax1, keta_mat, "Ketamine", HEATMAP_CMAP, vmin, vmax),
        (ax2, diff_mat, "Ketamine − Awake", HEATMAP_DIFF_CMAP, -diff_lim, diff_lim),
    ]:
        ax.imshow(
            mat,
            aspect="auto",
            origin="upper",
            extent=extent,
            cmap=cmap,
            vmin=vm_min,
            vmax=vm_max,
            interpolation="nearest",
        )
        ax.axvline(0, color="white", linestyle="--", linewidth=3, alpha=0.9)
        ax.axvline(
            STIM_DURATION_MS, color="white", linestyle="--", linewidth=3, alpha=0.9
        )
        for b in boundaries:
            ax.axhline(b, color="white", linestyle="--", linewidth=3)
        ax.set_title(panel_title)

    ax1.set_xlabel("Time from Onset (ms)")

    # shared colorbar to the right of keta panel
    sm_main = plt.cm.ScalarMappable(
        cmap=HEATMAP_CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    fig.colorbar(sm_main, cax=cax_main, label=ylabel)

    # diff colorbar to the right of diff panel
    cax_diff = make_axes_locatable(ax2).append_axes("right", size="8%", pad=0.15)
    sm_diff = plt.cm.ScalarMappable(
        cmap=HEATMAP_DIFF_CMAP, norm=plt.Normalize(vmin=-diff_lim, vmax=diff_lim)
    )
    fig.colorbar(sm_diff, cax=cax_diff, label=diff_ylabel)

    for ax in (ax0, ax1, ax2):
        ax.set_xlim(bin_centers[0], bin_centers[-1])

    # layer labels on leftmost data panel only, bold; y-axis label
    ax0.set_yticks([y for _, y in label_info])
    ax0.set_yticklabels([f"L{lyr}" for lyr, _ in label_info], fontweight="bold")
    ax0.set_ylabel("Neurons")
    ax1.set_yticks([])
    ax2.set_yticks([])

    _save(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_psth_heatmap(result: SessionResult, output_path: Path):
    """Three-panel PSTH heatmap for a single session (all neurons, not just responsive)."""
    ci = result.cluster_info.sort_values("brain_depth", ascending=True).reset_index(
        drop=True
    )

    awake_idx = {uid: i for i, uid in enumerate(result.awake_all.psth_unit_ids)}
    keta_idx = {uid: i for i, uid in enumerate(result.ketamine_all.psth_unit_ids)}

    awake_mat = result.awake_all.neuron_psths[
        [awake_idx[u] for u in ci["cluster_id"] if u in awake_idx]
    ]
    keta_mat = result.ketamine_all.neuron_psths[
        [keta_idx[u] for u in ci["cluster_id"] if u in keta_idx]
    ]

    common_uids = [u for u in ci["cluster_id"] if u in awake_idx and u in keta_idx]
    diff_mat = (
        result.ketamine_all.neuron_psths[[keta_idx[u] for u in common_uids]]
        - result.awake_all.neuron_psths[[awake_idx[u] for u in common_uids]]
    )

    _draw_psth_heatmap(
        awake_mat,
        keta_mat,
        diff_mat,
        ci,
        result.awake.bin_centers,
        result.session_name,
        output_path,
    )


def plot_session(result: SessionResult):
    """Save per-session activation curve, PSTH, and PSTH heatmap."""
    plot_activation_curve(
        result.awake.amplitude_stats,
        result.ketamine.amplitude_stats,
        result.output_dir / "activation_curve.pdf",
        title=result.session_name,
    )
    plot_psth(
        (result.awake.bin_centers, result.awake.psth, result.awake.psth_sem),
        (result.ketamine.bin_centers, result.ketamine.psth, result.ketamine.psth_sem),
        result.output_dir / "psth.pdf",
        title=result.session_name,
        awake_raster=result.awake.raster_spikes,
        keta_raster=result.ketamine.raster_spikes,
    )
    plot_psth_heatmap(result, result.output_dir / "psth_heatmap.pdf")


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
    pooled_dir = Path(config["files"].get("pooled_output_dir", "output_pooled"))
    pooled_dir.mkdir(exist_ok=True)

    print(
        f"{_TEAL}Calibration fit (degree-3 poly, V→mW/mm²): {np.poly1d(_cal_poly)}{_RESET}"
    )
    plot_calibration(pooled_dir / "calibration_fit.pdf")

    pooled = pool_sessions(results)
    stats_df = run_stats(
        pooled["awake_resp"],
        pooled["keta_resp"],
        pooled["n_neurons"],
        pooled["n_sessions"],
    )
    stats_df.to_csv(pooled_dir / "activation_curve_stats.csv", index=False)

    psth_stats_df = run_psth_stats(
        pooled["awake_psths"],
        pooled["keta_psths"],
        pooled["bin_centers"],
        pooled["n_neurons"],
    )
    psth_stats_df.to_csv(pooled_dir / "psth_stats.csv", index=False)

    pooled_title = (
        f"Pooled ({pooled['n_neurons']} neurons, {pooled['n_sessions']} sessions)"
    )
    plot_activation_curve(
        pooled["awake_stats"],
        pooled["keta_stats"],
        pooled_dir / "activation_curve_pooled.pdf",
        stats_df=stats_df,
        title=pooled_title,
    )
    plot_activation_curve(
        pooled["awake_stats"],
        None,
        pooled_dir / "activation_curve_pooled_awakeOnly.pdf",
        title=pooled_title,
        show_legend=False,
    )

    # global top-N raster neurons ranked by mean awake response in PULSE_WINDOW
    rp = pooled["raster_psths"]
    if len(rp):
        win_mask = (pooled["bin_centers"] >= PULSE_WINDOW[0]) & (
            pooled["bin_centers"] <= PULSE_WINDOW[1]
        )
        top_idx = np.argsort(rp[:, win_mask].mean(axis=1))[::-1][:RASTER_N_NEURONS]
        pooled_awake_raster = [pooled["raster_awake_spikes"][i] for i in top_idx]
        pooled_keta_raster = [pooled["raster_keta_spikes"][i] for i in top_idx]
    else:
        pooled_awake_raster = pooled_keta_raster = []

    plot_psth(
        (pooled["bin_centers"], pooled["awake_psth"], pooled["awake_psth_sem"]),
        (pooled["bin_centers"], pooled["keta_psth"], pooled["keta_psth_sem"]),
        pooled_dir / "psth_pooled.pdf",
        title=f"Pooled PSTH ({pooled['n_neurons']} neurons)",
        psth_stats_df=psth_stats_df,
        awake_raster=pooled_awake_raster,
        keta_raster=pooled_keta_raster,
    )

    # sort pooled neurons: group by layer (cortical order), then by depth within layer
    layer_order = ["1", "2/3", "4", "5", "6"]
    ci = pooled["heatmap_cluster_info"].copy()
    ci["_layer_rank"] = (
        ci["layer"]
        .map({lyr: i for i, lyr in enumerate(layer_order)})
        .fillna(len(layer_order))
    )
    ci_sorted = ci.sort_values(["_layer_rank", "brain_depth"], ascending=True).drop(
        columns="_layer_rank"
    )
    sort_idx = ci_sorted.index.values
    ci_sorted = ci_sorted.reset_index(drop=True)
    awake_hm = pooled["heatmap_awake"][sort_idx]
    keta_hm = pooled["heatmap_keta"][sort_idx]
    _draw_psth_heatmap(
        awake_hm,
        keta_hm,
        keta_hm - awake_hm,
        ci_sorted,
        pooled["bin_centers"],
        pooled_title,
        pooled_dir / "psth_heatmap_pooled.pdf",
    )

    sig_amps = stats_df.loc[stats_df["significant"], "amplitude"].tolist()
    print(
        f"{_TEAL}\nPooled: {pooled['n_neurons']} neurons across {pooled['n_sessions']} sessions{_RESET}"
    )
    print(f"{_TEAL}Significant amplitudes (FDR α={ALPHA}): {sig_amps}{_RESET}")


if __name__ == "__main__":
    main()
