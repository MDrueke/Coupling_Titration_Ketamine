"""activation titration analysis pipeline.

processes each session for two brain states (awake, ketamine), produces per-session
and pooled activation curves and pSTHs, and tests for state differences.
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

PULSE_WINDOW = (5.0, 35.0)  # ms post-onset for response measurement
BASELINE_EXCLUSION = (
    -100.0,
    500.0,
)  # ms: exclude this window around each pulse from baseline
# None = include all areas; list of layer strings from area_depths.csv (leading "L" stripped)
#  cortical: "1", "2/3", "4", "5", "6" — subcortical: "Ca1", "Th"
_CORTEX_LAYERS = [
    "1",
    "2/3",
    "4",
    "5",
    "6",
]  # always used for responsive PSTH (not user-configurable)

ZSCORE = False
MIN_FIRING_RATE_HZ = 0.05  # minimum firing rate in the awake window to include a neuron

PSTH_WINDOW = (100.0, 500.0)  # (pre_ms, post_ms) for all PSTHs
PSTH_WINDOW_RESPONSIVE = (10.0, 50.0)  # (pre_ms, post_ms) for responsive-only plots
PSTH_BIN_SIZE = 2.5  # ms
PSTH_AMPLITUDE_RANGE = (1, 5)  # (min, max) V — pulses used for PSTH
STIM_DURATION_MS = 28.0
INTERPOLATE_ARTIFACT = True  # interpolate PSTHs across stimulation artifacts
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
RASTER_N_NEURONS = 1  # top-N most responsive neurons shown in raster
RASTER_N_TRIALS = 30  # trials per neuron in raster (at highest stim amplitude)
RASTER_MIN_SPIKES = 1  # min mean spike count in stim period to be eligible for raster
# physical axes dimensions shared by both PSTH plots (inches)
PSTH_AX_W = 4.5
PSTH_AX_H = 2.0
PSTH_LAYER_OFFSET = 0.5  # vertical offset between layer traces in the layer PSTH plot
PSTH_LAYER_CMAP = "Wistia"  # colormap for layer traces; also controls tick label colors
PSTH_RASTER_H = 2 * PSTH_AX_H  # raster panel height (2:1 ratio relative to PSTH)
PSTH_RASTER_GAP = 0.15  # gap between raster and PSTH panels, inches
_PSTH_M = 0.1  # minimal figure margin; bbox_inches='tight' includes labels

ALPHA = 0.05  # FDR threshold for Wilcoxon + B-H correction

COLOR_AWAKE = "#D6604D"  # orange-red

ANESTHESIA_COLORS = {
    "ketamine": "#4393C3",  # blue
    "isoflurane": "#f5d442",  # yellow
    "urethane": "#5aa340",  # green
}

# colors for Ca/Th extra PSTH traces (fixed per area, independent of anesthetic)
COLOR_CA_AWAKE = "#F4A582"
COLOR_CA_ANESTH = "#669e46"
COLOR_TH_AWAKE = "#9957a1"
COLOR_TH_ANESTH = "#57a19c"

# additional area groups shown as separate traces in PSTH plots
PSTH_AREA_GROUPS = {
    "Ca1": {
        "layers": ["Ca1"],
        "color_awake": COLOR_CA_AWAKE,
        "color_anesth": COLOR_CA_ANESTH,
    },
    "Th": {
        "layers": ["Th"],
        "color_awake": COLOR_TH_AWAKE,
        "color_anesth": COLOR_TH_ANESTH,
    },
}

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
DARK_MODE = True

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
    "figure.figsize": (4, 2.5),
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

_DARK_BG = "#1a1a1a"
NATURE_STYLE_DARK = {
    **NATURE_STYLE,
    "axes.edgecolor": "white",
    "axes.facecolor": _DARK_BG,
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",
    "figure.facecolor": _DARK_BG,
    "grid.color": "#444444",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "legend.facecolor": _DARK_BG,
    "legend.edgecolor": "white",
}

_ACTIVE_STYLE = NATURE_STYLE_DARK if DARK_MODE else NATURE_STYLE

# derived colors that depend on dark mode
_STIM_COLOR = "#404040" if DARK_MODE else "lightgray"
_ZERO_COLOR = "#888888" if DARK_MODE else "gray"
_SIG_COLOR = "white" if DARK_MODE else "black"
_FOREGROUND_COLOR = "white" if DARK_MODE else "black"


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
    area_psths: Optional[dict] = (
        None  # {area_name: {"awake": (bc, psth, sem, neuron_psths), "keta": ...}}
    )
    layer_psths: Optional[dict] = (
        None  # {layer: {"awake": (bc, psth, sem, neuron_psths), "keta": ...}}
    )
    n_responsive_awake: int = 0
    n_responsive_keta: int = 0
    anesthesia_state: str = "ketamine"
    per_amp_psths: Optional[dict] = None  # {amp_v: {"awake": (bc,psth,sem,neuron_psths,raster), "keta": ...}}


def load_config(config_path: str = "config.toml") -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def create_output_dir(recording_dir: Path, config: dict) -> Path:
    output_dir = recording_dir / config["files"]["output_dir"]
    output_dir.mkdir(exist_ok=True)
    return output_dir


def filter_neurons(
    rec: Recording, layer_filter: Optional[List[str]], min_firing_rate_hz: float
) -> List[int]:
    """filter neurons by layer and minimum firing rate in the awake window."""
    if not rec.stateTimes:
        raise ValueError(
            "rec.stateTimes is empty — check meta.txt for awake/ketamine entries"
        )

    df = rec.clusterInfo.copy()

    if layer_filter is not None:
        df = df[df["layer"].isin(layer_filter)]
    n_after_layer = len(df)
    if layer_filter is not None:
        print(
            f"{_TEAL}\t...Layer filter {layer_filter}: {n_after_layer} neurons{_RESET}"
        )
    layer_counts_pre = df["layer"].value_counts().sort_index()
    print(
        f"{_TEAL}\t...Layer counts before FR filter: { {k: int(v) for k, v in layer_counts_pre.items()} }{_RESET}"
    )

    # firing rate filter: ≥ min_firing_rate_hz in the awake window only
    awake_start_s, awake_end_s_min = rec.stateTimes["awake"]
    awake_start_s *= 60
    awake_end_s = np.inf if np.isinf(awake_end_s_min) else awake_end_s_min * 60

    kept = []
    for uid in df["cluster_id"]:
        spikes = rec.unitSpikes[uid]
        if np.isinf(awake_end_s):
            awake_spikes = spikes[spikes >= awake_start_s]
            duration_s = spikes[-1] - awake_start_s if len(spikes) else 0.0
        else:
            awake_spikes = spikes[(spikes >= awake_start_s) & (spikes < awake_end_s)]
            duration_s = awake_end_s - awake_start_s
        fr = len(awake_spikes) / duration_s if duration_s > 0 else 0.0
        if fr >= min_firing_rate_hz:
            kept.append(uid)

    unit_ids = kept
    layer_counts_post = (
        df[df["cluster_id"].isin(unit_ids)]["layer"].value_counts().sort_index()
    )
    print(
        f"{_TEAL}\t...Layer counts after FR filter:  { {k: int(v) for k, v in layer_counts_post.items()} }{_RESET}"
    )
    print(
        f"{_TEAL}\t...{len(unit_ids)}/{n_after_layer} neurons passed FR filter "
        f"(≥{min_firing_rate_hz} Hz awake){_RESET}"
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
    baseline firing rate within the state's time window, excluding pulse periods.

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
    """return (bin_edges_s, bin_centers_ms, artifact_idx_list) for the global PSTH parameters."""
    pre_s = PSTH_WINDOW[0] / 1000
    post_s = PSTH_WINDOW[1] / 1000
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
    """apply artifact interpolation and Gaussian smoothing to a trial-averaged trace."""
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
    per-neuron per-amplitude response from preprocessed psth trace.
    Uses the same artifact interpolation and smoothing as calculate_psth.
    """
    bin_edges, bin_centers_ms, artifact_idx_list = _build_psth_bins()
    pre_s = PSTH_WINDOW[0] / 1000
    post_s = PSTH_WINDOW[1] / 1000
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
    """average per neuron per amplitude, then compute mean ± SEM across neurons."""
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
    returns (bin_centers_ms, psth_mean, psth_sem, neuron_psths, psth_unit_ids).
    neuron_psths has shape (n_neurons, n_bins) and is kept for pooling.
    """
    bin_edges, bin_centers, artifact_idx_list = _build_psth_bins()
    pre_s = PSTH_WINDOW[0] / 1000
    post_s = PSTH_WINDOW[1] / 1000
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


def _zeta_count(rec, unit_ids: List[int], onsets: np.ndarray, amps: np.ndarray) -> int:
    """count neurons responsive to a single state's onsets via ZETA test."""
    import logging

    from zetapy import zetatest

    mask = amps > ZETA_MIN_AMPLITUDE
    filtered = onsets[mask] if mask.any() else onsets
    if len(filtered) == 0:
        return 0
    dur = ZETA_MAX_DUR_MS / 1000.0
    _root_level = logging.root.level
    logging.root.setLevel(logging.CRITICAL)
    try:
        result = sum(
            zetatest(rec.unitSpikes[uid], filtered, dblUseMaxDur=dur)[0] <= ALPHA
            for uid in unit_ids
        )
    finally:
        logging.root.setLevel(_root_level)
    return result


def identify_responsive_neurons_zeta(
    rec,
    unit_ids: List[int],
    awake_onsets: np.ndarray,
    awake_amps: np.ndarray,
    keta_onsets: np.ndarray,
    keta_amps: np.ndarray,
) -> List[int]:
    """run ZETA test using trials from both states combined; returns unit_ids with p <= ALPHA."""
    from zetapy import zetatest

    mask_awake = awake_amps > ZETA_MIN_AMPLITUDE
    mask_keta = keta_amps > ZETA_MIN_AMPLITUDE
    onsets = np.sort(np.concatenate([awake_onsets[mask_awake], keta_onsets[mask_keta]]))
    if len(onsets) == 0:
        print(
            f"  [ZETA] No trials with amplitude > {ZETA_MIN_AMPLITUDE} V — falling back to all trials"
        )
        onsets = np.sort(np.concatenate([awake_onsets, keta_onsets]))

    import logging

    dur = ZETA_MAX_DUR_MS / 1000.0
    _root_level = logging.root.level
    logging.root.setLevel(logging.CRITICAL)
    try:
        result = [
            uid
            for uid in unit_ids
            if zetatest(rec.unitSpikes[uid], onsets, dblUseMaxDur=dur)[0] <= ALPHA
        ]
    finally:
        logging.root.setLevel(_root_level)
    return result


def identify_responsive_neurons(
    unit_ids: List[int],
    psth_unit_ids: List[int],
    neuron_psths: np.ndarray,
    baseline_stats: Dict,
    bin_centers: np.ndarray,
) -> List[int]:
    """
    return unit_ids whose mean z-scored response in PULSE_WINDOW exceeds
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
    """return a copy of StateData restricted to the given unit_ids / unique_ids."""
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
    """full analysis pipeline for one brain state."""
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


def _collect_raster_spikes(
    rec, unit_ids: List[int], onsets: np.ndarray
) -> List[List[np.ndarray]]:
    """per-trial spike times (ms re. onset) for each unit at the given onsets."""
    pre_s = PSTH_WINDOW[0] / 1000
    post_s = PSTH_WINDOW[1] / 1000
    result = []
    for uid in unit_ids:
        spikes = rec.unitSpikes[uid]
        trials = []
        for onset in onsets[:RASTER_N_TRIALS]:
            mask = (spikes >= onset - pre_s) & (spikes < onset + post_s)
            trials.append((spikes[mask] - onset) * 1000)
        result.append(trials)
    return result


def _compute_per_amp_psths(
    rec, unit_ids, unique_ids, awake_onsets, awake_amps,
    keta_onsets, keta_amps, raster_uids, baseline_stats
) -> dict:
    """compute PSTH + raster for each unique amplitude in PSTH_AMPLITUDE_RANGE."""
    amps_in_range = np.unique(
        awake_amps[(awake_amps >= PSTH_AMPLITUDE_RANGE[0]) & (awake_amps <= PSTH_AMPLITUDE_RANGE[1])]
    )
    per_amp = {}
    for amp in amps_in_range:
        a_mask = np.isclose(awake_amps, amp)
        k_mask = np.isclose(keta_amps, amp)
        a_onsets = awake_onsets[a_mask]
        k_onsets = keta_onsets[k_mask]
        if len(a_onsets) == 0 and len(k_onsets) == 0:
            continue
        # single-amplitude amps array so calculate_psth accepts it
        a_amps = np.full(len(a_onsets), amp)
        k_amps = np.full(len(k_onsets), amp)
        try:
            a_bc, a_psth, a_sem, a_mats, _ = calculate_psth(
                rec, unit_ids, a_onsets, a_amps, baseline_stats, ZSCORE
            )
            k_bc, k_psth, k_sem, k_mats, _ = calculate_psth(
                rec, unit_ids, k_onsets, k_amps, baseline_stats, ZSCORE
            )
        except ValueError:
            continue
        a_raster = _collect_raster_spikes(rec, raster_uids, a_onsets)
        k_raster = _collect_raster_spikes(rec, raster_uids, k_onsets)
        per_amp[float(amp)] = {
            "awake": (a_bc, a_psth, a_sem, a_mats, a_raster),
            "keta":  (k_bc, k_psth, k_sem, k_mats, k_raster),
        }
    return per_amp


def process_session(session_dir: Path, config: dict) -> SessionResult:
    """full pipeline for one session. Returns SessionResult."""
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
    anesthesia_state = stim_df.loc[
        stim_df["brain_state"] != "awake", "brain_state"
    ].iloc[0]
    awake_df = stim_df[stim_df["brain_state"] == "awake"]
    keta_df = stim_df[stim_df["brain_state"] == anesthesia_state]
    awake_onsets = awake_df["onset_time_s"].values
    awake_amps = awake_df["amplitude_v"].values
    keta_onsets = keta_df["onset_time_s"].values
    keta_amps = keta_df["amplitude_v"].values
    print(
        f"{_TEAL}  {len(awake_onsets)} awake pulses | {len(keta_onsets)} {anesthesia_state} pulses{_RESET}"
    )

    # 2. Load recording and filter neurons
    print("\n[2/3] Loading recording...")
    rec = Recording(recording_dir, config)
    unit_ids = filter_neurons(rec, _CORTEX_LAYERS, MIN_FIRING_RATE_HZ)
    unique_ids = [f"{session_name}_{uid}" for uid in unit_ids]
    print(f"{_TEAL}  {len(unit_ids)} neurons included{_RESET}")

    # 3. Analyse each state (all neurons passing layer + spike filter)
    print("\n[3/3] Computing responses...")
    print("  — Awake")
    awake_data_all = process_state(
        rec, unit_ids, unique_ids, awake_onsets, awake_amps, rec.stateTimes["awake"]
    )
    print(f"  — {anesthesia_state.title()}")
    keta_data_all = process_state(
        rec,
        unit_ids,
        unique_ids,
        keta_onsets,
        keta_amps,
        rec.stateTimes[anesthesia_state],
    )

    # cluster_info for heatmap always covers all neurons (before responsiveness filter)
    cluster_info = (
        rec.clusterInfo[rec.clusterInfo["cluster_id"].isin(unit_ids)][
            ["cluster_id", "brain_depth", "layer"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    # restrict psth/activation-curve data to the shared responsive set (same neurons in both
    # states, required for the paired Wilcoxon signed-rank test)
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
            if ZSCORE
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
        responsive_ids = unit_ids
        awake_data = awake_data_all
        keta_data = keta_data_all

    # per-state responsive counts for boxplot — layer filter only, no spike count minimum
    unit_ids_unfiltered = filter_neurons(rec, _CORTEX_LAYERS, 0)
    if RESPONSIVE_NEURON_DETECTION == "zeta":
        n_responsive_awake = _zeta_count(
            rec, unit_ids_unfiltered, awake_onsets, awake_amps
        )
        n_responsive_keta = _zeta_count(
            rec, unit_ids_unfiltered, keta_onsets, keta_amps
        )
    elif RESPONSIVE_ZSCORE_THRESHOLD is not None:
        n_responsive_awake = len(
            identify_responsive_neurons(
                unit_ids,
                awake_data_all.psth_unit_ids,
                awake_data_all.neuron_psths,
                awake_baseline_stats,
                awake_data_all.bin_centers,
            )
        )
        keta_baseline_stats = (
            None
            if ZSCORE
            else calculate_baseline_stats(
                rec,
                unit_ids,
                keta_onsets,
                BASELINE_EXCLUSION,
                rec.stateTimes[anesthesia_state],
            )
        )
        n_responsive_keta = len(
            identify_responsive_neurons(
                unit_ids,
                keta_data_all.psth_unit_ids,
                keta_data_all.neuron_psths,
                keta_baseline_stats,
                keta_data_all.bin_centers,
            )
        )
    else:
        n_responsive_awake = n_responsive_keta = len(responsive_ids)

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

    awake_data.amplitude_stats.to_csv(
        output_dir / "amplitude_response_awake.csv", index=False
    )
    keta_data.amplitude_stats.to_csv(
        output_dir / f"amplitude_response_{anesthesia_state}.csv", index=False
    )

    print(f"\n  Activation thresholds ({session_name}):")
    find_activation_threshold(
        awake_data.responses_df,
        "awake",
        output_path=output_dir / "threshold_awake.csv",
    )
    find_activation_threshold(
        keta_data.responses_df,
        anesthesia_state,
        output_path=output_dir / f"threshold_{anesthesia_state}.csv",
    )

    # compute separate PSTHs for each PSTH_AREA_GROUPS entry (no responsiveness filter)
    area_psths = {}
    for area_name, grp in PSTH_AREA_GROUPS.items():
        area_ids = filter_neurons(rec, grp["layers"], MIN_FIRING_RATE_HZ)
        if not area_ids:
            continue
        area_unique = [f"{session_name}_{uid}" for uid in area_ids]
        a_state = process_state(
            rec,
            area_ids,
            area_unique,
            awake_onsets,
            awake_amps,
            rec.stateTimes["awake"],
        )
        k_state = process_state(
            rec,
            area_ids,
            area_unique,
            keta_onsets,
            keta_amps,
            rec.stateTimes[anesthesia_state],
        )
        area_psths[area_name] = {
            "awake": (
                a_state.bin_centers,
                a_state.psth,
                a_state.psth_sem,
                a_state.neuron_psths,
            ),
            "keta": (
                k_state.bin_centers,
                k_state.psth,
                k_state.psth_sem,
                k_state.neuron_psths,
            ),
        }
        print(f"{_TEAL}  Area '{area_name}': {len(area_ids)} neurons{_RESET}")

    # compute per-cortical-layer PSTHs (no responsiveness filter)
    layer_psths = {}
    for layer_name in _CORTEX_LAYERS:
        layer_ids = filter_neurons(rec, [layer_name], MIN_FIRING_RATE_HZ)
        if not layer_ids:
            continue
        layer_unique = [f"{session_name}_{uid}" for uid in layer_ids]
        a_state = process_state(
            rec,
            layer_ids,
            layer_unique,
            awake_onsets,
            awake_amps,
            rec.stateTimes["awake"],
        )
        k_state = process_state(
            rec,
            layer_ids,
            layer_unique,
            keta_onsets,
            keta_amps,
            rec.stateTimes[anesthesia_state],
        )
        layer_psths[layer_name] = {
            "awake": (
                a_state.bin_centers,
                a_state.psth,
                a_state.psth_sem,
                a_state.neuron_psths,
            ),
            "keta": (
                k_state.bin_centers,
                k_state.psth,
                k_state.psth_sem,
                k_state.neuron_psths,
            ),
        }
        print(f"{_TEAL}  Layer '{layer_name}': {len(layer_ids)} neurons{_RESET}")

    per_amp_psths = _compute_per_amp_psths(
        rec, unit_ids, unique_ids, awake_onsets, awake_amps,
        keta_onsets, keta_amps, raster_uids, baseline_stats=None,
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
        area_psths=area_psths,
        layer_psths=layer_psths,
        n_responsive_awake=n_responsive_awake,
        n_responsive_keta=n_responsive_keta,
        anesthesia_state=anesthesia_state,
        per_amp_psths=per_amp_psths,
    )


def pool_sessions(results: List[SessionResult]) -> dict:
    """concatenate per-neuron data across sessions and recompute group-level summaries."""
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

    # pool all-neuron (non-responsive) PSTHs for the "all areas" PSTH plot
    all_awake_psths = np.vstack([r.awake_all.neuron_psths for r in results])
    all_keta_psths = np.vstack([r.ketamine_all.neuron_psths for r in results])
    n_all = len(all_awake_psths)

    # pool area group PSTHs across sessions
    pooled_area_psths = {}
    for area_name in PSTH_AREA_GROUPS:
        a_mats = [
            r.area_psths[area_name]["awake"][3]
            for r in results
            if r.area_psths and area_name in r.area_psths
        ]
        k_mats = [
            r.area_psths[area_name]["keta"][3]
            for r in results
            if r.area_psths and area_name in r.area_psths
        ]
        if not a_mats:
            continue
        a_all = np.vstack(a_mats)
        k_all = np.vstack(k_mats)
        na = len(a_all)
        nk = len(k_all)
        pooled_area_psths[area_name] = {
            "awake": (
                bin_centers,
                np.mean(a_all, axis=0),
                np.std(a_all, axis=0) / np.sqrt(na),
            ),
            "keta": (
                bin_centers,
                np.mean(k_all, axis=0),
                np.std(k_all, axis=0) / np.sqrt(nk),
            ),
        }

    # pool per-layer PSTHs across sessions
    pooled_layer_psths = {}
    for layer_name in _CORTEX_LAYERS:
        a_mats = [
            r.layer_psths[layer_name]["awake"][3]
            for r in results
            if r.layer_psths and layer_name in r.layer_psths
        ]
        k_mats = [
            r.layer_psths[layer_name]["keta"][3]
            for r in results
            if r.layer_psths and layer_name in r.layer_psths
        ]
        if not a_mats:
            continue
        a_all = np.vstack(a_mats)
        k_all = np.vstack(k_mats)
        na, nk = len(a_all), len(k_all)
        pooled_layer_psths[layer_name] = {
            "awake": (
                bin_centers,
                np.mean(a_all, axis=0),
                np.std(a_all, axis=0) / np.sqrt(na),
            ),
            "keta": (
                bin_centers,
                np.mean(k_all, axis=0),
                np.std(k_all, axis=0) / np.sqrt(nk),
            ),
        }

    # pool per-amplitude PSTHs across sessions
    all_amps = sorted({amp for r in results if r.per_amp_psths for amp in r.per_amp_psths})
    pooled_per_amp = {}
    for amp in all_amps:
        a_mats = [r.per_amp_psths[amp]["awake"][3] for r in results
                  if r.per_amp_psths and amp in r.per_amp_psths]
        k_mats = [r.per_amp_psths[amp]["keta"][3] for r in results
                  if r.per_amp_psths and amp in r.per_amp_psths]
        a_rasters = [spike for r in results if r.per_amp_psths and amp in r.per_amp_psths
                     for spike in r.per_amp_psths[amp]["awake"][4]]
        k_rasters = [spike for r in results if r.per_amp_psths and amp in r.per_amp_psths
                     for spike in r.per_amp_psths[amp]["keta"][4]]
        if not a_mats:
            continue
        a_all = np.vstack(a_mats)
        k_all = np.vstack(k_mats)
        na, nk = len(a_all), len(k_all)
        pooled_per_amp[amp] = {
            "awake": (bin_centers, np.mean(a_all, axis=0), np.std(a_all, axis=0) / np.sqrt(na), a_all, a_rasters),
            "keta":  (bin_centers, np.mean(k_all, axis=0), np.std(k_all, axis=0) / np.sqrt(nk), k_all, k_rasters),
        }

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
        "area_psths": pooled_area_psths,
        "layer_psths": pooled_layer_psths,
        "awake_all_psth": np.mean(all_awake_psths, axis=0),
        "awake_all_psth_sem": np.std(all_awake_psths, axis=0) / np.sqrt(n_all),
        "keta_all_psth": np.mean(all_keta_psths, axis=0),
        "keta_all_psth_sem": np.std(all_keta_psths, axis=0) / np.sqrt(n_all),
        "n_neurons_all": n_all,
        "per_amp_psths": pooled_per_amp,
    }


def _bh_correction(p_values: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """benjamini-Hochberg FDR correction. Returns boolean reject array."""
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
    wilcoxon signed-rank test (per amplitude) on per-neuron mean responses.
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
    """wilcoxon signed-rank test on per-neuron mean response in PULSE_WINDOW."""
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


_VOLT_TO_MW = {float(k): float(v) for k, v in VOLTAGE_TO_mW.items()}
# fit calibration curve on non-zero-output points (sub-threshold points excluded)
_cal_pairs = sorted((v, mw) for v, mw in _VOLT_TO_MW.items() if mw > 0)
_cal_v = np.array([v for v, mw in _cal_pairs])
_cal_mw = np.array([mw for v, mw in _cal_pairs])
_cal_poly = np.polyfit(_cal_v, _cal_mw, 3)


def _volt_to_mw(v: float) -> float:
    """convert voltage (V) to mW/mm² using a degree-3 polynomial fit of the calibration data."""
    return float(max(0.0, np.polyval(_cal_poly, float(v))))


def find_activation_threshold(
    responses_df: pd.DataFrame,
    label: str,
    output_path: Optional[Path] = None,
) -> dict:
    """
    estimate activation threshold two ways:
      1. wilcoxon: lowest amplitude in the first run of ≥2 consecutive BH-significant
         one-sample tests (response > 0 across neurons).
      2. breakpoint: piecewise-linear (hockey-stick) fit to the population mean curve;
         threshold = bend point.
    prints both estimates in V and mW/mm².
    """
    from scipy.optimize import curve_fit

    per_neuron = (
        responses_df.groupby(["unique_id", "amplitude"])["response"]
        .mean()
        .reset_index()
    )
    amps = np.array(sorted(per_neuron["amplitude"].unique()))

    rows = []
    for amp in amps:
        vals = per_neuron[per_neuron["amplitude"] == amp]["response"].values
        if len(vals) < 5:
            rows.append(
                {
                    "amplitude": amp,
                    "p_value": 1.0,
                    "n": len(vals),
                    "median_response": float(np.median(vals)),
                }
            )
            continue
        _, p = scipy.stats.wilcoxon(vals, alternative="greater")
        rows.append(
            {
                "amplitude": amp,
                "p_value": p,
                "n": len(vals),
                "median_response": float(np.median(vals)),
            }
        )

    df = pd.DataFrame(rows)
    df["significant"] = _bh_correction(df["p_value"].values)

    # threshold = first amplitude where this and the next amplitude are both significant
    sig = df["significant"].values
    threshold_wilcoxon_v = None
    for i in range(len(sig) - 1):
        if sig[i] and sig[i + 1]:
            threshold_wilcoxon_v = float(df["amplitude"].iloc[i])
            break

    # piecewise linear (hockey-stick) fit on mW scale: response = slope * max(x - x0, 0)
    mean_resp = np.array(
        [per_neuron[per_neuron["amplitude"] == amp]["response"].mean() for amp in amps]
    )
    mw = np.array([_volt_to_mw(a) for a in amps])

    threshold_bp_mw = None
    try:

        def _hockey(x, x0, slope):
            return slope * np.maximum(x - x0, 0)

        p0 = [
            mw[len(mw) // 3],
            (mean_resp[-1] - mean_resp[0]) / max(mw[-1] - mw[0], 1e-9),
        ]
        popt, _ = curve_fit(_hockey, mw, mean_resp, p0=p0, maxfev=5000)
        threshold_bp_mw = float(np.clip(popt[0], mw[0], mw[-1]))
    except Exception:
        pass

    threshold_wilcoxon_mw = (
        _volt_to_mw(threshold_wilcoxon_v) if threshold_wilcoxon_v is not None else None
    )

    if threshold_wilcoxon_v is not None:
        print(
            f"{_TEAL}  [{label}] threshold (wilcoxon): {threshold_wilcoxon_v:.3f} V / {threshold_wilcoxon_mw:.1f} mW/mm²{_RESET}"
        )
    else:
        print(f"{_TEAL}  [{label}] threshold (wilcoxon): not found{_RESET}")
    if threshold_bp_mw is not None:
        print(
            f"{_TEAL}  [{label}] threshold (breakpoint fit): {threshold_bp_mw:.1f} mW/mm²{_RESET}"
        )
    else:
        print(f"{_TEAL}  [{label}] threshold (breakpoint fit): fit failed{_RESET}")

    df["amplitude_mw"] = df["amplitude"].map(_volt_to_mw)
    if output_path is not None:
        df.to_csv(output_path, index=False)

    return {
        "threshold_wilcoxon_v": threshold_wilcoxon_v,
        "threshold_wilcoxon_mw": threshold_wilcoxon_mw,
        "threshold_bp_mw": threshold_bp_mw,
        "table": df,
    }


def _save(output_path: Path, **kwargs):
    """save current figure as both .pdf and .png with tight bounding box."""
    kwargs.setdefault("bbox_inches", "tight")

    plt.savefig(output_path.with_suffix(".pdf"), transparent=True, **kwargs)
    plt.savefig(output_path.with_suffix(".png"), transparent=True, **kwargs)


def plot_calibration(output_path: Path):
    """scatter of calibration points + polynomial fit trace."""
    plt.rcParams.update(_ACTIVE_STYLE)
    fig, ax = plt.subplots(figsize=(6, 4))

    all_v = np.array(sorted(_VOLT_TO_MW))
    all_mw = np.array([_VOLT_TO_MW[v] for v in all_v])
    ax.scatter(
        all_v, all_mw, color=_FOREGROUND_COLOR, zorder=3, label="Calibration points"
    )

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
    threshold_mw: Optional[float] = None,
    keta_color: str = ANESTHESIA_COLORS["ketamine"],
    keta_label: str = "Ketamine",
    also_save_log: bool = False,
):
    """overlay awake (orange) and optionally anesthesia activation curves."""
    plt.rcParams.update(_ACTIVE_STYLE)
    fig, ax = plt.subplots()

    pairs = [(awake_stats, COLOR_AWAKE, "Awake")]
    if keta_stats is not None:
        pairs.append((keta_stats, keta_color, keta_label))
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

    if stats_df is not None:
        sig_mw = np.sort([_volt_to_mw(a) for a in stats_df.loc[stats_df["significant"], "amplitude"]])
        if len(sig_mw):
            all_mw = np.sort(np.unique(awake_stats["amplitude"].map(_volt_to_mw).values))
            half_step = np.diff(all_mw).min() / 2 if len(all_mw) > 1 else 1.0
            # group runs of significant intensities that are adjacent in the tested set
            sig_idx = [int(np.argmin(np.abs(all_mw - v))) for v in sig_mw]
            runs, run = [], [sig_idx[0]]
            for pi, ci in zip(sig_idx[:-1], sig_idx[1:]):
                if ci == pi + 1:
                    run.append(ci)
                else:
                    runs.append(run); run = [ci]
            runs.append(run)
            for run in runs:
                ax.hlines(1.04, all_mw[run[0]] - half_step, all_mw[run[-1]] + half_step,
                          colors=_SIG_COLOR, linewidth=2,
                          transform=ax.get_xaxis_transform(), clip_on=False)

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
    if threshold_mw is not None:
        ax.axvline(
            threshold_mw,
            color=_FOREGROUND_COLOR,
            linestyle="--",
            linewidth=1.5,
            label="_nolegend_",
        )
        if show_legend:
            ax.legend(frameon=False)
    plt.tight_layout()
    _save(output_path, dpi=300)
    if also_save_log:
        ax.set_xscale("log")
        ax.set_xticks([])  # let matplotlib choose log ticks
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        log_path = output_path.with_name(output_path.stem + "_log" + output_path.suffix)
        _save(log_path, dpi=300)
    plt.close()


def plot_psth(
    awake_psth_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    keta_psth_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    output_path: Path,
    title: str = "PSTH",
    psth_stats_df: Optional[pd.DataFrame] = None,
    awake_raster: Optional[List] = None,
    keta_raster: Optional[List] = None,
    extra_traces: Optional[List[dict]] = None,
    normalize: bool = False,
    legend_outside: bool = False,
    window=None,
    keta_color: str = ANESTHESIA_COLORS["ketamine"],
    keta_label: str = "Ketamine",
):
    """overlay awake and anesthesia PSTHs, optionally with a raster panel above."""
    if window is None:
        window = PSTH_WINDOW

    def _norm(psth, sem, bc):
        """subtract pre-stim median, divide by peak."""
        pre = psth[bc < 0]
        corrected = psth - (np.median(pre) if len(pre) else 0.0)
        peak = corrected.max()
        if peak <= 0:
            return corrected, sem
        return corrected / peak, sem / peak

    plt.rcParams.update(_ACTIVE_STYLE)
    has_raster = awake_raster is not None and len(awake_raster) > 0

    ax_w = PSTH_AX_W / 2 if window is PSTH_WINDOW_RESPONSIVE else PSTH_AX_W
    fig_w = _PSTH_M + ax_w + _PSTH_M
    if has_raster:
        fig_h = _PSTH_M + PSTH_AX_H + PSTH_RASTER_GAP + PSTH_RASTER_H + _PSTH_M
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes(
            [
                _PSTH_M / fig_w,
                _PSTH_M / fig_h,
                ax_w / fig_w,
                PSTH_AX_H / fig_h,
            ]
        )
        raster_b = (_PSTH_M + PSTH_AX_H + PSTH_RASTER_GAP) / fig_h
        ax_raster = fig.add_axes(
            [
                _PSTH_M / fig_w,
                raster_b,
                ax_w / fig_w,
                PSTH_RASTER_H / fig_h,
            ]
        )
    else:
        fig_h = _PSTH_M + PSTH_AX_H + _PSTH_M
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes(
            [
                _PSTH_M / fig_w,
                _PSTH_M / fig_h,
                ax_w / fig_w,
                PSTH_AX_H / fig_h,
            ]
        )

    bin_centers, awake_psth, awake_sem = awake_psth_data
    _, keta_psth, keta_sem = keta_psth_data
    if normalize:
        awake_psth, awake_sem = _norm(awake_psth, awake_sem, bin_centers)
        keta_psth, keta_sem = _norm(keta_psth, keta_sem, bin_centers)

    # precompute extra trace display data (normalized if needed) — used for plotting and bounds
    extra_plot = []  # (bc, psth, sem, color, label_str)
    if extra_traces:
        for tr in extra_traces:
            for state, color_key, suffix in [
                ("awake", "color_awake", " (Awake)"),
                ("keta", "color_anesth", f" ({keta_label})"),
            ]:
                bc_e, psth_e, sem_e = tr[state]
                if normalize:
                    psth_e, sem_e = _norm(psth_e, sem_e, bc_e)
                extra_plot.append(
                    (bc_e, psth_e, sem_e, tr[color_key], f"{tr['label']}{suffix}")
                )

    # compute y bounds from all traces including SEM
    all_upper = [
        float((awake_psth + awake_sem).max()),
        float((keta_psth + keta_sem).max()),
    ]
    all_lower = [
        float((awake_psth - awake_sem).min()),
        float((keta_psth - keta_sem).min()),
    ]
    for _, p, s, _, _ in extra_plot:
        all_upper.append(float((p + s).max()))
        all_lower.append(float((p - s).min()))
    data_ymax = max(all_upper)
    data_ymin = min(all_lower)
    y_range = max(data_ymax - data_ymin, 1e-9)
    y_pad = 0.05 * y_range
    ylim_bottom = data_ymin - y_pad
    ylim_top = data_ymax + y_pad

    ax.axvspan(0, STIM_DURATION_MS, color=_STIM_COLOR, alpha=0.3, zorder=0)
    ax.axvline(0, color=_ZERO_COLOR, linewidth=1, alpha=0.5)

    for bc_e, psth_e, sem_e, c, lbl in extra_plot:
        ax.plot(bc_e, psth_e, color=c, linewidth=2, label=lbl, zorder=2)
        ax.fill_between(
            bc_e,
            psth_e - sem_e,
            psth_e + sem_e,
            color=c,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )

    for psth, sem, color, label, zo in [
        (awake_psth, awake_sem, COLOR_AWAKE, "Awake", 4),
        (keta_psth, keta_sem, keta_color, keta_label, 6),
    ]:
        ax.plot(bin_centers, psth, color=color, linewidth=2, label=label, zorder=zo)
        ax.fill_between(
            bin_centers,
            psth - sem,
            psth + sem,
            color=color,
            alpha=0.2,
            linewidth=0,
            zorder=zo - 1,
        )

    has_sig_bars = False
    bar_y = data_ymax + y_pad
    if psth_stats_df is not None:
        sig_bins = psth_stats_df[psth_stats_df["significant"]].sort_values("bin_start")
        if len(sig_bins):
            has_sig_bars = True
            ylim_top = bar_y + y_pad * 3
            # group consecutive significant bins into runs; draw one line per run
            starts = sig_bins["bin_start"].values
            ends = sig_bins["bin_end"].values
            run_start, run_end = starts[0], ends[0]
            for s, e in zip(starts[1:], ends[1:]):
                if np.isclose(s, run_end):
                    run_end = e
                else:
                    ax.hlines(bar_y, run_start, run_end, colors=_SIG_COLOR, linewidth=2)
                    run_start, run_end = s, e
            ax.hlines(bar_y, run_start, run_end, colors=_SIG_COLOR, linewidth=2)

    xmin, xmax = -window[0], window[1]
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Time from Onset (ms)")
    if normalize:
        ax.set_ylabel("Norm. Δ FR (a.u.)")
    elif ZSCORE:
        ax.set_ylabel("Z-scored Firing Rate")
    else:
        ax.set_ylabel("Firing Rate (Hz)")

    if has_raster:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], color=COLOR_AWAKE, linewidth=4, label="Awake"),
            Line2D([0], [0], color=keta_color, linewidth=4, label=keta_label),
        ]
        if extra_traces:
            for tr in extra_traces:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=tr["color_awake"],
                        linewidth=2,
                        label=f"{tr['label']} (Awake)",
                    )
                )
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=tr["color_anesth"],
                        linewidth=2,
                        label=f"{tr['label']} ({keta_label})",
                    )
                )
        if legend_outside:
            ax_raster.legend(
                handles=handles,
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                borderaxespad=0,
            )
        else:
            ax_raster.legend(handles=handles, frameon=True, loc="upper right")
    else:
        if legend_outside:
            ax.legend(
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                borderaxespad=0,
            )
        else:
            ax.legend(frameon=True, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if normalize:
        yticks = list(range(int(np.floor(data_ymin)), int(np.ceil(data_ymax)) + 1))
    elif ZSCORE:
        top_tick = int(np.ceil(data_ymax / 100) * 100)
        yticks = [0, top_tick // 2, top_tick]
    else:
        yticks = [0, int(round(data_ymax))]
        ylim_bottom = 0
        if not has_sig_bars:
            ylim_top = data_ymax  # actual max including SEM
    ax.set_yticks(yticks)
    ax.set_ylim(ylim_bottom, ylim_top)

    if has_raster:
        n = RASTER_N_TRIALS
        neuron_gap = 1
        N = len(awake_raster)
        keta_offset = N * (n + neuron_gap)

        ax_raster.axvspan(0, STIM_DURATION_MS, color=_STIM_COLOR, alpha=0.3, zorder=0)
        ax_raster.axvline(0, color=_ZERO_COLOR, linewidth=1, alpha=0.5)
        for neuron_i in range(N):
            awake_y_base = neuron_i * (n + neuron_gap)
            keta_y_base = keta_offset + neuron_i * (n + neuron_gap)
            for trial_j, spikes in enumerate(awake_raster[neuron_i]):
                y = awake_y_base + trial_j
                ax_raster.vlines(spikes, y, y + 0.85, color=COLOR_AWAKE, linewidth=1.0)
            for trial_j, spikes in enumerate(keta_raster[neuron_i]):
                y = keta_y_base + trial_j
                ax_raster.vlines(spikes, y, y + 0.85, color=keta_color, linewidth=1.0)

        total_y = keta_offset + N * (n + neuron_gap)
        ax_raster.set_ylim(-0.5, total_y)
        ax_raster.set_yticks([0, total_y - 1])
        ax_raster.set_ylabel("ex. Neuron trial #", labelpad=4)
        ax_raster.spines[["right", "top", "bottom"]].set_visible(False)
        ax_raster.spines["left"].set_visible(True)
        ax_raster.set_xlim(xmin, xmax)
        ax_raster.set_xticks([])

    ax.set_xticks([0, int(xmax // 2), int(xmax)])
    ax.set_xticklabels(["0", str(int(xmax // 2)), str(int(xmax))])

    _save(output_path, dpi=300)
    plt.close()


def _get_layer_boundaries(layers: List[str]):
    """return (boundary_positions, [(layer_name, midpoint_y), ...]) from an ordered list."""
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
    anesthesia_label: str = "Anesthesia",
):
    """shared renderer for three-panel PSTH heatmap: awake | anesthesia | difference."""
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.rcParams.update(_ACTIVE_STYLE)
    plt.rcParams.update(HEATMAP_FONT_SCALE)
    if DARK_MODE:
        plt.rcParams.update(
            {
                "text.color": "white",
                "axes.labelcolor": "white",
                "axes.titlecolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
            }
        )

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
        (ax1, keta_mat, anesthesia_label, HEATMAP_CMAP, vmin, vmax),
        (
            ax2,
            diff_mat,
            f"{anesthesia_label} − Awake",
            HEATMAP_DIFF_CMAP,
            -diff_lim,
            diff_lim,
        ),
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

    sm_main = plt.cm.ScalarMappable(
        cmap=HEATMAP_CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    fig.colorbar(sm_main, cax=cax_main, label=ylabel)

    cax_diff = make_axes_locatable(ax2).append_axes("right", size="8%", pad=0.15)
    sm_diff = plt.cm.ScalarMappable(
        cmap=HEATMAP_DIFF_CMAP, norm=plt.Normalize(vmin=-diff_lim, vmax=diff_lim)
    )
    fig.colorbar(sm_diff, cax=cax_diff, label=diff_ylabel)

    for ax in (ax0, ax1, ax2):
        ax.set_xlim(bin_centers[0], bin_centers[-1])

    ax0.set_yticks([y for _, y in label_info])
    ax0.set_yticklabels([f"L{lyr}" for lyr, _ in label_info], fontweight="bold")
    ax0.set_ylabel("Neurons")
    ax1.set_yticks([])
    ax2.set_yticks([])

    _save(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_psth_heatmap(result: SessionResult, output_path: Path):
    """three-panel PSTH heatmap for a single session (all neurons, not just responsive)."""
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
        anesthesia_label=result.anesthesia_state.title(),
    )


def plot_responsive_counts(
    results: List["SessionResult"],
    output_path: Path,
    keta_color: str = ANESTHESIA_COLORS["ketamine"],
    keta_label: str = "Ketamine",
):
    """thin boxplot comparing n_responsive_awake vs n_responsive_anesthesia across sessions."""
    plt.rcParams.update(_ACTIVE_STYLE)

    awake_counts = [r.n_responsive_awake for r in results]
    keta_counts = [r.n_responsive_keta for r in results]

    fig_w = _PSTH_M + 1 + _PSTH_M
    fig_h = _PSTH_M + PSTH_AX_H + _PSTH_M
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes(
        [
            _PSTH_M / fig_w,
            _PSTH_M / fig_h,
            1 / fig_w,
            PSTH_AX_H / fig_h,
        ]
    )

    bp = ax.boxplot(
        [awake_counts, keta_counts],
        positions=[1, 2],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=_FOREGROUND_COLOR, linewidth=2),
        whiskerprops=dict(color=_FOREGROUND_COLOR, linewidth=1.5),
        capprops=dict(color=_FOREGROUND_COLOR, linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], [COLOR_AWAKE, keta_color]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_linewidth(0)

    rng = np.random.default_rng(0)
    for x_center, counts, color in [
        (1, awake_counts, COLOR_AWAKE),
        (2, keta_counts, keta_color),
    ]:
        jitter = rng.uniform(-0.08, 0.08, len(counts))
        ax.scatter(x_center + jitter, counts, color=color, s=80, zorder=5, alpha=1.0)

    # mann-whitney u test
    from scipy.stats import mannwhitneyu

    _, p = mannwhitneyu(awake_counts, keta_counts, alternative="two-sided")
    if p < 0.001:
        sig_label = "***"
    elif p < 0.01:
        sig_label = "**"
    elif p < 0.05:
        sig_label = "*"
    else:
        sig_label = "ns"

    # y limits based on whisker extents
    whisker_vals = [w.get_ydata()[1] for w in bp["whiskers"]]
    y_lo = min(whisker_vals)
    y_hi = max(whisker_vals)
    span = max(y_hi - y_lo, 1)
    ax.set_ylim(y_lo - 0.1 * span, y_hi + 0.3 * span)

    # significance bracket between the two boxes
    bracket_y = y_hi + 0.15 * span
    ax.annotate(
        "",
        xy=(2, bracket_y),
        xytext=(1, bracket_y),
        arrowprops=dict(arrowstyle="-", color=_FOREGROUND_COLOR, lw=1.2),
    )
    ax.text(
        1.5,
        bracket_y,
        sig_label,
        ha="center",
        va="bottom",
        fontsize=12,
        color=_FOREGROUND_COLOR,
    )

    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Awake", keta_label], rotation=45, ha="right")
    ax.set_ylabel("Responsive neurons (n)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    _save(output_path, dpi=300)
    plt.close()


def plot_psth_layers(
    layer_psths: Optional[dict], output_path: Path, state: str = "awake"
):
    """waterfall PSTH for one brain state: offset trace per cortical layer, colored by PSTH_LAYER_CMAP.

    state: "awake" or "keta"
    """
    if not layer_psths:
        return

    def _norm(psth, sem, bc):
        pre = psth[bc < 0]
        corrected = psth - (np.median(pre) if len(pre) else 0.0)
        peak = corrected.max()
        if peak <= 0:
            return corrected, sem
        return corrected / peak, sem / peak

    plt.rcParams.update(_ACTIVE_STYLE)

    layer_order = ["6", "5", "4", "2/3", "1"]  # bottom to top
    present = [l for l in layer_order if l in layer_psths]
    n = len(present)
    if n == 0:
        return

    cmap = plt.colormaps[PSTH_LAYER_CMAP]
    colors = {lyr: cmap(i / (n - 1) if n > 1 else 0.5) for i, lyr in enumerate(present)}

    fig_w = _PSTH_M + PSTH_AX_W + _PSTH_M
    fig_h = _PSTH_M + PSTH_AX_H + _PSTH_M
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes(
        [
            _PSTH_M / fig_w,
            _PSTH_M / fig_h,
            PSTH_AX_W / fig_w,
            PSTH_AX_H / fig_h,
        ]
    )

    ax.axvspan(0, STIM_DURATION_MS, color=_STIM_COLOR, alpha=0.3, zorder=0)
    ax.axvline(0, color=_ZERO_COLOR, linewidth=1, alpha=0.5)

    ytick_pos = []
    for i, lyr in enumerate(present):
        offset = i * PSTH_LAYER_OFFSET
        bc, psth, sem = layer_psths[lyr][state][:3]
        psth, sem = _norm(psth, sem, bc)
        c = colors[lyr]
        zo = n - i  # layer 6 (i=0) gets highest zorder
        ax.fill_between(
            bc,
            psth + offset - sem,
            psth + offset + sem,
            color=c,
            alpha=0.15,
            linewidth=0,
            zorder=zo - 0.5,
        )
        ax.plot(bc, psth + offset, color=c, linewidth=2, zorder=zo)
        ytick_pos.append(offset)

    xmin, xmax = -PSTH_WINDOW[0], PSTH_WINDOW[1]
    ax.set_xlim(xmin, xmax)
    ax.set_xticks([0, int(xmax // 2), int(xmax)])
    ax.set_xticklabels(["0", str(int(xmax // 2)), str(int(xmax))])
    ax.set_xlabel("Time from Onset (ms)")
    ax.set_ylabel("Layer")
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([f"L{lyr}" for lyr in present])
    ax.set_ylim(ytick_pos[0] - 0.15, ytick_pos[-1] + 1.1)

    for tick, lyr in zip(ax.yaxis.get_ticklabels(), present):
        tick.set_color(colors[lyr])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(output_path, dpi=300)
    plt.close()


def _build_extra_traces(
    area_psths: Optional[dict], area_filter: Optional[List[str]] = None
) -> Optional[List[dict]]:
    """convert area_psths dict to extra_traces list for plot_psth."""
    if not area_psths:
        return None
    traces = []
    for area_name, grp in PSTH_AREA_GROUPS.items():
        if area_name not in area_psths:
            continue
        if area_filter is not None and area_name not in area_filter:
            continue
        entry = area_psths[area_name]
        traces.append(
            {
                "label": area_name,
                "color_awake": grp["color_awake"],
                "color_anesth": grp["color_anesth"],
                "awake": entry["awake"][:3],
                "keta": entry["keta"][:3],
            }
        )
    return traces or None


def plot_psth_heatmap_responsive_comparison(
    results: List["SessionResult"],
    output_path: Path,
    anesthesia_label: str = "Anesthesia",
):
    """pooled heatmap of responsive neurons: awake | keta side by side in one imshow.

    Neurons sorted top→bottom by descending mean awake firing rate in PULSE_WINDOW.
    X-axis covers PSTH_WINDOW_RESPONSIVE with duplicated tick labels for each half.
    Only neurons present in both awake and keta psth_unit_ids are included.
    """
    plt.rcParams.update(_ACTIVE_STYLE)

    # collect per-session matrices for neurons common to both states
    all_awake, all_keta = [], []
    bin_centers = None
    for r in results:
        a_map = {uid: i for i, uid in enumerate(r.awake.psth_unit_ids)}
        k_map = {uid: i for i, uid in enumerate(r.ketamine.psth_unit_ids)}
        common = [uid for uid in r.awake.psth_unit_ids if uid in k_map]
        if not common:
            continue
        a_rows = [a_map[uid] for uid in common]
        k_rows = [k_map[uid] for uid in common]
        all_awake.append(r.awake.neuron_psths[a_rows])
        all_keta.append(r.ketamine.neuron_psths[k_rows])
        if bin_centers is None:
            bin_centers = r.awake.bin_centers

    if not all_awake:
        return

    awake_mat = np.concatenate(all_awake, axis=0)  # (N, n_bins)
    keta_mat = np.concatenate(all_keta, axis=0)

    # slice to responsive time window
    pre, post = PSTH_WINDOW_RESPONSIVE
    t_mask = (bin_centers >= -pre) & (bin_centers <= post)
    bc = bin_centers[t_mask]
    awake_mat = awake_mat[:, t_mask]
    keta_mat = keta_mat[:, t_mask]

    # sort by mean awake firing rate in PULSE_WINDOW (descending = highest at top)
    win_mask = (bc >= PULSE_WINDOW[0]) & (bc <= PULSE_WINDOW[1])
    sort_key = awake_mat[:, win_mask].mean(axis=1)
    order = np.argsort(sort_key)[::-1]
    awake_mat = awake_mat[order]
    keta_mat = keta_mat[order]

    N, n_bins = awake_mat.shape
    combined = np.concatenate([awake_mat, keta_mat], axis=1)  # (N, 2*n_bins)

    vmin = 0.0
    vmax = np.percentile(combined, 99)

    ax_w = PSTH_AX_W / 2
    ax_h = max(2.0, min(N * 0.02, 8.0))
    cbar_w = 0.15
    cbar_gap = 0.05
    fig_w = _PSTH_M + ax_w + cbar_gap + cbar_w + _PSTH_M
    fig_h = _PSTH_M + ax_h + _PSTH_M
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([_PSTH_M / fig_w, _PSTH_M / fig_h, ax_w / fig_w, ax_h / fig_h])
    cax = fig.add_axes(
        [
            (_PSTH_M + ax_w + cbar_gap) / fig_w,
            _PSTH_M / fig_h,
            cbar_w / fig_w,
            ax_h / fig_h,
        ]
    )

    cmap = plt.colormaps["viridis"].copy()
    im = ax.imshow(
        combined,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=[-0.5, 2 * n_bins - 0.5, N - 0.5, -0.5],
    )

    # white separator between awake and keta halves
    sep_color = "white" if DARK_MODE else "white"
    ax.axvline(n_bins - 0.5, color=sep_color, linewidth=2, zorder=5)

    # "Awake" / "Ketamine" labels above each half
    label_y = -0.5 - N * 0.04
    ax.text(
        n_bins / 2 - 0.5,
        label_y,
        "Awake",
        ha="center",
        va="bottom",
        fontsize=12,
        color=_FOREGROUND_COLOR,
    )
    ax.text(
        n_bins * 1.5 - 0.5,
        label_y,
        anesthesia_label,
        ha="center",
        va="bottom",
        fontsize=12,
        color=_FOREGROUND_COLOR,
    )

    # x-axis ticks and stimulus marker lines — edit tick_vals to change ticks/lines
    tick_vals = [0, 28]  # ms: stimulus onset and offset
    tick_positions = []
    tick_labels_list = []
    for half_offset in (0, n_bins):
        for t in tick_vals:
            col = np.argmin(np.abs(bc - t)) + half_offset
            tick_positions.append(col)
            tick_labels_list.append(str(int(t)))
        # dashed vertical lines marking stimulus on/off for this half
        for t in tick_vals:
            col = np.argmin(np.abs(bc - t)) + half_offset
            ax.axvline(col, color="white", linewidth=0.8, linestyle="--", zorder=4)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_list)
    ax.set_xlabel("Time from Onset (ms)")

    ax.set_yticks([])
    ax.set_ylabel("Neurons")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Firing Rate (Hz)")
    if DARK_MODE:
        cb.ax.yaxis.set_tick_params(color="white")
        cb.outline.set_edgecolor("white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    _save(output_path, dpi=300)
    plt.close()


def plot_session(result: SessionResult):
    """save per-session activation curve, PSTH, and PSTH heatmap."""
    ac = result.anesthesia_state
    ac_color = ANESTHESIA_COLORS.get(ac, ANESTHESIA_COLORS["ketamine"])
    ac_label = ac.title()

    plot_activation_curve(
        result.awake.amplitude_stats,
        result.ketamine.amplitude_stats,
        result.output_dir / "activation_curve.pdf",
        title=result.session_name,
        keta_color=ac_color,
        keta_label=ac_label,
    )
    # plot 1: responsive cortical neurons only
    plot_psth(
        (result.awake.bin_centers, result.awake.psth, result.awake.psth_sem),
        (result.ketamine.bin_centers, result.ketamine.psth, result.ketamine.psth_sem),
        result.output_dir / "psth_responsive.pdf",
        title=result.session_name,
        awake_raster=result.awake.raster_spikes,
        keta_raster=result.ketamine.raster_spikes,
        window=PSTH_WINDOW_RESPONSIVE,
        keta_color=ac_color,
        keta_label=ac_label,
    )
    # plot 2a/b/c: all cortical neurons, with optional extra area traces
    _awake_all_data = (
        result.awake_all.bin_centers,
        result.awake_all.psth,
        result.awake_all.psth_sem,
    )
    _keta_all_data = (
        result.ketamine_all.bin_centers,
        result.ketamine_all.psth,
        result.ketamine_all.psth_sem,
    )
    _psth_all_kwargs = dict(
        title=result.session_name,
        normalize=True,
        legend_outside=True,
        keta_color=ac_color,
        keta_label=ac_label,
    )
    plot_psth(
        _awake_all_data,
        _keta_all_data,
        result.output_dir / "psth_all_cortex.pdf",
        **_psth_all_kwargs,
    )
    for area in ("Th", "Ca1"):
        if result.area_psths and area in result.area_psths:
            plot_psth(
                _awake_all_data,
                _keta_all_data,
                result.output_dir / f"psth_all_{area}.pdf",
                extra_traces=_build_extra_traces(result.area_psths, [area]),
                **_psth_all_kwargs,
            )
    plot_psth_layers(
        result.layer_psths, result.output_dir / "psth_layers_awake.pdf", state="awake"
    )
    plot_psth_layers(
        result.layer_psths, result.output_dir / f"psth_layers_{ac}.pdf", state="keta"
    )
    plot_psth_heatmap(result, result.output_dir / "psth_heatmap.pdf")

    # per-amplitude responsive PSTH
    if result.per_amp_psths:
        for amp, data in result.per_amp_psths.items():
            amp_tag = f"{amp:g}V"
            a_bc, a_psth, a_sem, _, a_raster = data["awake"]
            k_bc, k_psth, k_sem, _, k_raster = data["keta"]
            plot_psth(
                (a_bc, a_psth, a_sem),
                (k_bc, k_psth, k_sem),
                result.output_dir / f"psth_responsive_{amp_tag}.pdf",
                title=f"{result.session_name} — {amp_tag}",
                awake_raster=a_raster,
                keta_raster=k_raster,
                window=PSTH_WINDOW_RESPONSIVE,
                keta_color=ac_color,
                keta_label=ac_label,
            )


def main():
    config = load_config()
    with open("sessions.toml", "rb") as f:
        sessions_cfg = tomllib.load(f)

    results = []
    for d in sessions_cfg["sessions"]["dirs"]:
        result = process_session(Path(d), config)
        plot_session(result)
        results.append(result)

    base_pooled_dir = Path(config["files"].get("pooled_output_dir", "output_pooled"))
    base_pooled_dir.mkdir(exist_ok=True)

    # group sessions by anesthetic; pool and plot each group separately
    by_anesthetic = {}
    for r in results:
        by_anesthetic.setdefault(r.anesthesia_state, []).append(r)

    for anesthetic, group in by_anesthetic.items():
        ac_color = ANESTHESIA_COLORS.get(anesthetic, ANESTHESIA_COLORS["ketamine"])
        ac_label = anesthetic.title()

        pooled_dir = base_pooled_dir / anesthetic
        pooled_dir.mkdir(exist_ok=True)

        plot_responsive_counts(
            group,
            pooled_dir / "responsive_counts.pdf",
            keta_color=ac_color,
            keta_label=ac_label,
        )

        print(
            f"{_TEAL}Calibration fit (degree-3 poly, V→mW/mm²): {np.poly1d(_cal_poly)}{_RESET}"
        )
        plot_calibration(pooled_dir / "calibration_fit.pdf")

        pooled = pool_sessions(group)
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

        print(f"\n  Activation thresholds (pooled, {anesthetic}):")
        awake_thresh = find_activation_threshold(
            pooled["awake_resp"],
            "awake",
            output_path=pooled_dir / "threshold_awake.csv",
        )
        keta_thresh = find_activation_threshold(
            pooled["keta_resp"],
            anesthetic,
            output_path=pooled_dir / f"threshold_{anesthetic}.csv",
        )

        pooled_title = f"Pooled {ac_label} ({pooled['n_neurons']} neurons, {pooled['n_sessions']} sessions)"
        plot_activation_curve(
            pooled["awake_stats"],
            pooled["keta_stats"],
            pooled_dir / "activation_curve_pooled.pdf",
            stats_df=stats_df,
            title=pooled_title,
            threshold_mw=awake_thresh["threshold_bp_mw"],
            keta_color=ac_color,
            keta_label=ac_label,
            also_save_log=True,
        )
        plot_activation_curve(
            pooled["awake_stats"],
            None,
            pooled_dir / "activation_curve_pooled_awakeOnly.pdf",
            title=pooled_title,
            show_legend=False,
            threshold_mw=awake_thresh["threshold_bp_mw"],
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

        # plot 1: responsive cortical neurons only
        plot_psth(
            (pooled["bin_centers"], pooled["awake_psth"], pooled["awake_psth_sem"]),
            (pooled["bin_centers"], pooled["keta_psth"], pooled["keta_psth_sem"]),
            pooled_dir / "psth_pooled_responsive.pdf",
            title=f"Pooled PSTH ({pooled['n_neurons']} responsive neurons)",
            psth_stats_df=psth_stats_df,
            awake_raster=pooled_awake_raster,
            keta_raster=pooled_keta_raster,
            window=PSTH_WINDOW_RESPONSIVE,
            keta_color=ac_color,
            keta_label=ac_label,
        )
        # plot 2a/b/c: all cortical neurons, with optional extra area traces
        _awake_all_data = (
            pooled["bin_centers"],
            pooled["awake_all_psth"],
            pooled["awake_all_psth_sem"],
        )
        _keta_all_data = (
            pooled["bin_centers"],
            pooled["keta_all_psth"],
            pooled["keta_all_psth_sem"],
        )
        _psth_all_kwargs = dict(
            title=f"Pooled PSTH all areas ({pooled['n_neurons_all']} neurons)",
            normalize=True,
            legend_outside=True,
            keta_color=ac_color,
            keta_label=ac_label,
        )
        plot_psth(
            _awake_all_data,
            _keta_all_data,
            pooled_dir / "psth_pooled_all_cortex.pdf",
            **_psth_all_kwargs,
        )
        for area in ("Th", "Ca1"):
            if pooled.get("area_psths") and area in pooled["area_psths"]:
                plot_psth(
                    _awake_all_data,
                    _keta_all_data,
                    pooled_dir / f"psth_pooled_all_{area}.pdf",
                    extra_traces=_build_extra_traces(pooled["area_psths"], [area]),
                    **_psth_all_kwargs,
                )
        plot_psth_layers(
            pooled.get("layer_psths"),
            pooled_dir / "psth_pooled_layers_awake.pdf",
            state="awake",
        )
        plot_psth_layers(
            pooled.get("layer_psths"),
            pooled_dir / f"psth_pooled_layers_{anesthetic}.pdf",
            state="keta",
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
            anesthesia_label=ac_label,
        )

        plot_psth_heatmap_responsive_comparison(
            group,
            pooled_dir / "psth_heatmap_responsive_comparison.pdf",
            anesthesia_label=ac_label,
        )

        # pooled per-amplitude responsive PSTHs
        for amp, data in pooled.get("per_amp_psths", {}).items():
            amp_tag = f"{amp:g}V"
            a_bc, a_psth, a_sem, _, a_raster = data["awake"]
            k_bc, k_psth, k_sem, _, k_raster = data["keta"]
            plot_psth(
                (a_bc, a_psth, a_sem),
                (k_bc, k_psth, k_sem),
                pooled_dir / f"psth_pooled_responsive_{amp_tag}.pdf",
                title=f"Pooled PSTH — {amp_tag} ({pooled['n_neurons']} neurons)",
                awake_raster=a_raster,
                keta_raster=k_raster,
                window=PSTH_WINDOW_RESPONSIVE,
                keta_color=ac_color,
                keta_label=ac_label,
            )

        sig_amps = stats_df.loc[stats_df["significant"], "amplitude"].tolist()
        print(
            f"{_TEAL}\nPooled [{anesthetic}]: {pooled['n_neurons']} neurons across {pooled['n_sessions']} sessions{_RESET}"
        )
        print(f"{_TEAL}Significant amplitudes (FDR α={ALPHA}): {sig_amps}{_RESET}")


if __name__ == "__main__":
    main()
