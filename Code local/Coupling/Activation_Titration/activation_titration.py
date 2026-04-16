"""activation titration analysis pipeline.

processes each session for two brain states (awake, ketamine), produces per-session
and pooled activation curves and pSTHs, and tests for state differences.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.stats
import tomllib

_TEAL = "\033[38;2;187;230;228m"
_RESET = "\033[0m"

from recording import Recording, resolve_session_paths

# ── neuron filtering ──────────────────────────────────────────────────────────
_CORTEX_LAYERS = ["1", "2/3", "4", "5", "6"]  # fixed; used for all cortical analyses
MIN_FIRING_RATE_HZ = 0.05  # min awake-window FR to include a neuron

# ── stimulus & response windows ───────────────────────────────────────────────
STIM_DURATION_MS = 28.0
PULSE_WINDOW = (5.0, 35.0)  # ms post-onset for per-trial response measurement
BASELINE_EXCLUSION = (-100.0, 500.0)  # ms: exclude around each pulse from baseline FR
INTERPOLATE_ARTIFACT = True
ARTIFACT_WINDOWS_MS = [(0.0, 4.5), (27.0, 32.0)]  # ms post-onset

# ── responsiveness detection ──────────────────────────────────────────────────
RESPONSIVE_NEURON_DETECTION = "zeta"  # "zscore" or "zeta"
RESPONSIVE_ZSCORE_THRESHOLD = 4.0  # Only activate when RESPONSIVE_NEURON_DETECTION = "zscore"; min mean z-score in PULSE_WINDOW (awake); None to disable
ZETA_RESPONSIVE_AMPLITUDES: Optional[List[float]] = [
    7.0,
    60.0,
]  # [min, max] mW for initial ZETA; None = all
ZETA_MAX_DUR_MS = 35.0  # ZETA analysis window (ms)

# ── PSTH computation ──────────────────────────────────────────────────────────
ZSCORE = False
PSTH_WINDOW = (200.0, 500.0)  # (pre_ms, post_ms) for all-neuron PSTHs
PSTH_WINDOW_RESPONSIVE = (10.0, 50.0)  # (pre_ms, post_ms) for responsive-only plots
PSTH_BIN_SIZE = 2.5  # ms
PSTH_SMOOTH_SIGMA_MS = 2  # gaussian smoothing sigma in ms; 0 or None to disable
PSTH_LAYER_SMOOTH_SIGMA_MS = (
    4  # extra smoothing applied to layer PSTH traces at plot time
)

# ── amplitude subsets ─────────────────────────────────────────────────────────
LAYER_PSTH_AMPLITUDES: Optional[List[float]] = [
    7.0,
    60.0,
]  # [min, max] mW for layer PSTH; None = all
POOLED_AMPLITUDES: Optional[List[float]] = [
    7.0,
    60.0,
]  # [min, max] mW for combined PSTH; None = skip

# ── raster ────────────────────────────────────────────────────────────────────
RASTER_N_NEURONS = 0  # top-N responsive neurons shown in raster
RASTER_N_TRIALS = 30  # trials per neuron (at highest amplitude)
RASTER_MIN_SPIKES = 0.5  # min mean spike count in stim period to be eligible

# ── PSTH display ─────────────────────────────────────────────────────────────
PSTH_SHOW_LEGEND = False  # show legend on PSTH plots

# ── figure layout (inches) ────────────────────────────────────────────────────
PSTH_AX_W = 4.5
PSTH_AX_H = 2.0
PSTH_RASTER_H = 2 * PSTH_AX_H  # raster panel height
PSTH_RASTER_GAP = 0.15  # gap between raster and PSTH panels
PSTH_LAYER_OFFSET = 0.5  # vertical offset between layer traces
PSTH_LAYER_CMAP = "Wistia"  # colormap for layer traces
_PSTH_M = 0.1  # minimal figure margin

# ── statistics ────────────────────────────────────────────────────────────────
ALPHA = 0.05  # FDR threshold for Wilcoxon + B-H correction

# ── response probability ───────────────────────────────────────────────────────
RESP_PROB_K = 1.0          # threshold = baseline_count + K * baseline_std (Poisson)
RESP_PROB_THRESHOLD = 0.5  # min response probability to call a neuron's amplitude threshold

# ── parallelism ───────────────────────────────────────────────────────────────
N_WORKERS: Optional[int] = (
    18  # session worker processes; lower if running low on memory
)

# ── colors ────────────────────────────────────────────────────────────────────
COLOR_AWAKE = "#D6604D"
ANESTHESIA_COLORS = {
    "ketamine": "#4393C3",
    "isoflurane": "#f5d442",
    "urethane": "#5aa340",
}
COLOR_CA_AWAKE = "#F4A582"
COLOR_CA_ANESTH = "#669e46"
COLOR_TH_AWAKE = "#9957a1"
COLOR_TH_ANESTH = "#57a19c"

# ── area groups (extra traces in PSTH plots) ──────────────────────────────────
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

HEATMAP_CLIM_PERCENTILE = (
    98  # upper percentile for heatmap color limits (lower = 100 - this)
)
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
    resp_prob_df: pd.DataFrame  # columns: unique_id, amplitude, response_prob
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
    n_neurons_total: int = 0
    n_responsive_awake: int = 0
    n_responsive_keta: int = 0
    anesthesia_state: str = "ketamine"
    per_amp_psths: Optional[dict] = (
        None  # {amp_v: {"awake": (bc,psth,sem,neuron_psths,raster), "keta": ...}}
    )
    awake_firing_rates: Optional[List[float]] = (
        None  # per-neuron FR in awake window (Hz)
    )
    keta_firing_rates: Optional[List[float]] = (
        None  # per-neuron FR in anesthesia window (Hz)
    )
    awake_neuron_thresholds: Optional[List[float]] = (
        None  # per-neuron mW threshold where resp_prob >= RESP_PROB_THRESHOLD
    )
    keta_neuron_thresholds: Optional[List[float]] = None


def load_config(config_path: str = "config.toml") -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def create_output_dir(recording_dir: Path, config: dict) -> Path:
    output_dir = recording_dir / config["files"]["output_dir"]
    output_dir.mkdir(exist_ok=True)
    return output_dir


def filter_neurons(
    rec: Recording,
    layer_filter: Optional[List[str]],
    min_firing_rate_hz: float,
    verbose: bool = False,
) -> List[int]:
    """filter neurons by layer and minimum firing rate in the awake window."""
    if not rec.stateTimes:
        raise ValueError(
            "rec.stateTimes is empty — check meta.txt for awake/ketamine entries"
        )

    df = rec.clusterInfo.copy()
    if layer_filter is not None:
        df = df[df["layer"].isin(layer_filter)]

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

    if verbose:
        print(
            f"{_TEAL}\t...{len(kept)}/{len(df)} neurons passed filter "
            f"(layer={layer_filter}, FR≥{min_firing_rate_hz} Hz){_RESET}"
        )
    return kept


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


def calculate_response_probability(
    rec,
    unit_ids: List[int],
    unique_ids: List[str],
    pulse_onsets: np.ndarray,
    amplitudes: np.ndarray,
    baseline_stats: Dict,
) -> pd.DataFrame:
    """per-neuron per-amplitude response probability.

    A trial counts as a response if spike count in PULSE_WINDOW exceeds
    baseline_count + RESP_PROB_K * baseline_std (Poisson approximation).
    Returns DataFrame with columns [unique_id, amplitude, response_prob].
    """
    pre_s = PSTH_WINDOW[0] / 1000
    post_s = PSTH_WINDOW[1] / 1000
    win_lo = PULSE_WINDOW[0] / 1000
    win_hi = PULSE_WINDOW[1] / 1000
    win_dur = win_hi - win_lo

    unique_amps = np.unique(amplitudes)
    amp_onsets = {amp: pulse_onsets[np.isclose(amplitudes, amp)] for amp in unique_amps}

    rows = []
    for uid, uniq_id in zip(unit_ids, unique_ids):
        b = baseline_stats[uid]
        baseline_count = b["mean"] * win_dur
        baseline_std_count = np.sqrt(b["mean"] * win_dur)  # Poisson
        threshold_count = baseline_count + RESP_PROB_K * baseline_std_count
        spikes = rec.unitSpikes[uid]
        for amp, onsets in amp_onsets.items():
            trial_counts = []
            for onset in onsets:
                rel = spikes - onset
                n = int(np.sum((rel >= win_lo) & (rel < win_hi)))
                trial_counts.append(n)
            prob = float(np.mean(np.array(trial_counts) > threshold_count))
            rows.append({"unique_id": uniq_id, "amplitude": amp, "response_prob": prob})
    return pd.DataFrame(rows)


def neuron_thresholds_mw(resp_prob_df: pd.DataFrame) -> List[float]:
    """for each neuron, return the lowest mW amplitude where response_prob >= RESP_PROB_THRESHOLD.

    Neurons that never reach threshold are excluded (not set to NaN) so the
    distribution reflects only neurons that do respond.
    """
    thresholds = []
    for uid, grp in resp_prob_df.groupby("unique_id"):
        grp = grp.sort_values("amplitude")
        above = grp[grp["response_prob"] >= RESP_PROB_THRESHOLD]
        if len(above):
            thresholds.append(_volt_to_mw(above["amplitude"].iloc[0]))
    return thresholds


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
    all onsets/amplitudes passed in are used (filter before calling if needed).
    """
    bin_edges, bin_centers, artifact_idx_list = _build_psth_bins()
    pre_s = PSTH_WINDOW[0] / 1000
    post_s = PSTH_WINDOW[1] / 1000
    bin_s = PSTH_BIN_SIZE / 1000

    selected_onsets = pulse_onsets
    if len(selected_onsets) == 0:
        raise ValueError("No pulses provided")

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


def _available_cores() -> int:
    """physical core count minus 1 reserved for the OS.

    Uses psutil for an accurate physical count (ignores hyperthreading).
    Falls back to half of os.cpu_count() if psutil is not installed.
    """
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or os.cpu_count() or 2
    except ImportError:
        physical = max(2, (os.cpu_count() or 2) // 2)
    return max(1, physical - 1)


def _zeta_pvals(spike_arrays: List[np.ndarray], onsets: np.ndarray) -> List[float]:
    """run ZETA sequentially across neurons; boolParallel=False avoids nested Pool inside daemon workers."""
    import logging

    from zetapy import zetatest

    dur = ZETA_MAX_DUR_MS / 1000.0
    p_values = []
    for spikes in spike_arrays:
        _level = logging.root.level
        logging.root.setLevel(logging.CRITICAL)
        try:
            p_values.append(
                zetatest(spikes, onsets, dblUseMaxDur=dur, boolParallel=False)[0]
            )
        finally:
            logging.root.setLevel(_level)
    return p_values


def _zeta_count(rec, unit_ids: List[int], onsets: np.ndarray, amps: np.ndarray) -> int:
    """count neurons responsive to a single state's onsets via ZETA test."""
    if len(onsets) == 0 or not unit_ids:
        return 0
    p_values = _zeta_pvals([rec.unitSpikes[uid] for uid in unit_ids], onsets)
    return sum(p <= ALPHA for p in p_values)


def identify_responsive_neurons_zeta(
    rec,
    unit_ids: List[int],
    awake_onsets: np.ndarray,
    awake_amps: np.ndarray,
    keta_onsets: np.ndarray,
    keta_amps: np.ndarray,
) -> List[int]:
    """run ZETA test using trials from both states combined; returns unit_ids with p <= ALPHA.

    If ZETA_RESPONSIVE_AMPLITUDES is set, only trials within [min, max] mW are used.
    If None, all trials are used.
    """
    if ZETA_RESPONSIVE_AMPLITUDES is not None:
        lo_mw, hi_mw = ZETA_RESPONSIVE_AMPLITUDES
        a_mw = np.array([_volt_to_mw(v) for v in awake_amps])
        k_mw = np.array([_volt_to_mw(v) for v in keta_amps])
        mask_awake = (a_mw >= lo_mw) & (a_mw <= hi_mw)
        mask_keta = (k_mw >= lo_mw) & (k_mw <= hi_mw)
        onsets = np.sort(
            np.concatenate([awake_onsets[mask_awake], keta_onsets[mask_keta]])
        )
        if len(onsets) == 0:
            onsets = np.sort(np.concatenate([awake_onsets, keta_onsets]))
    else:
        onsets = np.sort(np.concatenate([awake_onsets, keta_onsets]))

    p_values = _zeta_pvals([rec.unitSpikes[uid] for uid in unit_ids], onsets)
    return [uid for uid, p in zip(unit_ids, p_values) if p <= ALPHA]


def identify_responsive_neurons(
    unit_ids: List[int],
    psth_unit_ids: List[int],
    neuron_psths: np.ndarray,
    baseline_stats: Dict,
    bin_centers: np.ndarray,
) -> List[int]:
    """
    return unit_ids whose mean z-scored response in PULSE_WINDOW exceeds
    RESPONSIVE_ZSCORE_THRESHOLD (awake data, all amplitudes).
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
    n_bins = state.neuron_psths.shape[1] if state.neuron_psths.ndim == 2 else 0
    if len(new_psths) == 0:
        new_psth = np.zeros(n_bins)
        new_sem = np.zeros(n_bins)
    else:
        new_psth = np.mean(new_psths, axis=0)
        new_sem = np.std(new_psths, axis=0) / np.sqrt(len(new_psths))

    # filter responses and recompute amplitude stats
    new_resp_df = state.responses_df[state.responses_df["unique_id"].isin(uniq_set)]
    new_amp_stats = aggregate_by_amplitude(new_resp_df)
    new_resp_prob_df = state.resp_prob_df[state.resp_prob_df["unique_id"].isin(uniq_set)]

    return StateData(
        pulse_onsets=state.pulse_onsets,
        amplitudes=state.amplitudes,
        responses_df=new_resp_df,
        amplitude_stats=new_amp_stats,
        resp_prob_df=new_resp_prob_df,
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
    # always compute baseline_stats — needed for both z-scoring and response probability
    baseline_stats = calculate_baseline_stats(
        rec, unit_ids, pulse_onsets, BASELINE_EXCLUSION, state_window
    )
    baseline_stats_for_psth = baseline_stats if ZSCORE else None

    responses_df = calculate_responses(
        rec,
        unit_ids,
        unique_ids,
        pulse_onsets,
        amplitudes,
        PULSE_WINDOW,
        baseline_stats_for_psth,
        ZSCORE,
    )
    amplitude_stats = aggregate_by_amplitude(responses_df)
    resp_prob_df = calculate_response_probability(
        rec, unit_ids, unique_ids, pulse_onsets, amplitudes, baseline_stats
    )
    bin_centers, psth, psth_sem, neuron_psths, psth_unit_ids = calculate_psth(
        rec,
        unit_ids,
        pulse_onsets,
        amplitudes,
        baseline_stats_for_psth,
        ZSCORE,
    )
    return StateData(
        pulse_onsets=pulse_onsets,
        amplitudes=amplitudes,
        responses_df=responses_df,
        amplitude_stats=amplitude_stats,
        resp_prob_df=resp_prob_df,
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


def _zeta_responsive_at_amp(rec, candidate_ids, a_onsets, k_onsets):
    """run ZETA using trials from both states at a single amplitude; return responsive IDs."""
    onsets = np.sort(np.concatenate([a_onsets, k_onsets]))
    if len(onsets) == 0 or not candidate_ids:
        return []
    p_values = _zeta_pvals([rec.unitSpikes[uid] for uid in candidate_ids], onsets)
    return [uid for uid, p in zip(candidate_ids, p_values) if p <= ALPHA]


def _compute_per_amp_psths(
    rec,
    candidate_ids,
    unique_ids,
    awake_onsets,
    awake_amps,
    keta_onsets,
    keta_amps,
    raster_uids,
    baseline_stats,
    all_unit_ids=None,
) -> dict:
    """compute PSTH + raster for each unique amplitude tested.

    ZETA (or zscore) responsiveness is re-evaluated per amplitude so each plot
    only shows neurons responsive at that specific intensity.
    """
    per_amp = {}
    for amp in np.unique(awake_amps):
        a_mask = np.isclose(awake_amps, amp)
        k_mask = np.isclose(keta_amps, amp)
        a_onsets = awake_onsets[a_mask]
        k_onsets = keta_onsets[k_mask]
        if len(a_onsets) == 0 and len(k_onsets) == 0:
            continue
        a_amps_arr = np.full(len(a_onsets), amp)
        k_amps_arr = np.full(len(k_onsets), amp)

        # per-amplitude responsiveness
        if RESPONSIVE_NEURON_DETECTION == "zeta":
            resp_ids = _zeta_responsive_at_amp(rec, candidate_ids, a_onsets, k_onsets)
        else:
            resp_ids = candidate_ids  # zscore path: reuse combined responsive set

        _all_cortex = filter_neurons(rec, _CORTEX_LAYERS, 0, verbose=False)
        n_resp_awake = (
            _zeta_count(rec, _all_cortex, a_onsets, a_amps_arr)
            if RESPONSIVE_NEURON_DETECTION == "zeta"
            else len(resp_ids)
        )
        n_resp_keta = (
            _zeta_count(rec, _all_cortex, k_onsets, k_amps_arr)
            if RESPONSIVE_NEURON_DETECTION == "zeta"
            else len(resp_ids)
        )

        if not resp_ids:
            continue
        try:
            a_bc, a_psth, a_sem, a_mats, _ = calculate_psth(
                rec, resp_ids, a_onsets, a_amps_arr, baseline_stats, ZSCORE
            )
            k_bc, k_psth, k_sem, k_mats, _ = calculate_psth(
                rec, resp_ids, k_onsets, k_amps_arr, baseline_stats, ZSCORE
            )
        except ValueError:
            continue

        # raster from the per-amp responsive set (top neuron by awake response)
        if RASTER_N_NEURONS > 0:
            win_mask = (a_bc >= PULSE_WINDOW[0]) & (a_bc <= PULSE_WINDOW[1])
            if len(resp_ids) > 0 and a_mats.shape[1] > 0:
                top_idx = int(np.argmax(a_mats[:, win_mask].mean(axis=1)))
                raster_ids_amp = [resp_ids[top_idx]]
            else:
                raster_ids_amp = raster_uids
            a_raster = _collect_raster_spikes(rec, raster_ids_amp, a_onsets)
            k_raster = _collect_raster_spikes(rec, raster_ids_amp, k_onsets)
        else:
            a_raster = []
            k_raster = []

        entry = {
            "awake": (a_bc, a_psth, a_sem, a_mats, a_raster),
            "keta": (k_bc, k_psth, k_sem, k_mats, k_raster),
            "n_responsive_awake": n_resp_awake,
            "n_responsive_keta": n_resp_keta,
        }
        if all_unit_ids is not None:
            try:
                aa_bc, aa_psth, aa_sem, aa_mats, _ = calculate_psth(
                    rec, all_unit_ids, a_onsets, a_amps_arr, baseline_stats, ZSCORE
                )
                ak_bc, ak_psth, ak_sem, ak_mats, _ = calculate_psth(
                    rec, all_unit_ids, k_onsets, k_amps_arr, baseline_stats, ZSCORE
                )
                entry["awake_all"] = (aa_bc, aa_psth, aa_sem, aa_mats)
                entry["keta_all"] = (ak_bc, ak_psth, ak_sem, ak_mats)
            except ValueError:
                pass
        per_amp[float(amp)] = entry
    return per_amp


def process_session(session_dir: Path, config: dict) -> SessionResult:
    """full pipeline for one session. Returns SessionResult."""
    session_dir = Path(session_dir)
    session_name = session_dir.name
    paths = resolve_session_paths(session_dir)
    recording_dir = paths["recording_dir"]
    output_dir = create_output_dir(recording_dir, config)

    # 1. Load pre-computed pulse table (produced by run_alignment.py)
    stim_file = output_dir / "stim_amplitudes.csv"
    if not stim_file.exists():
        raise FileNotFoundError(f"{stim_file} not found. Run run_alignment.py first.")
    stim_df = pd.read_csv(stim_file).dropna(subset=["onset_time_s", "amplitude_v"])
    anesthesia_state = stim_df.loc[
        stim_df["brain_state"] != "awake", "brain_state"
    ].iloc[0]
    awake_df = stim_df[stim_df["brain_state"] == "awake"]
    keta_df = stim_df[stim_df["brain_state"] == anesthesia_state]
    awake_onsets = awake_df["onset_time_s"].values
    awake_amps = awake_df["amplitude_v"].values
    keta_onsets = keta_df["onset_time_s"].values
    keta_amps = keta_df["amplitude_v"].values

    # 2. Load recording and filter neurons
    rec = Recording(recording_dir, config)
    unit_ids = filter_neurons(rec, _CORTEX_LAYERS, MIN_FIRING_RATE_HZ)
    n_neurons_total = len(unit_ids)
    unique_ids = [f"{session_name}_{uid}" for uid in unit_ids]

    # 3. Analyse each state (all neurons passing layer + spike filter)
    awake_data_all = process_state(
        rec, unit_ids, unique_ids, awake_onsets, awake_amps, rec.stateTimes["awake"]
    )
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

    # per-state responsive counts for boxplot — LAYER_PSTH_AMPLITUDES range, layer filter only
    unit_ids_unfiltered = filter_neurons(rec, _CORTEX_LAYERS, 0)
    if RESPONSIVE_NEURON_DETECTION == "zeta":
        if LAYER_PSTH_AMPLITUDES is not None:
            _bp_lo, _bp_hi = LAYER_PSTH_AMPLITUDES
            _a_mw = np.array([_volt_to_mw(v) for v in awake_amps])
            _k_mw = np.array([_volt_to_mw(v) for v in keta_amps])
            _bp_a_mask = (_a_mw >= _bp_lo) & (_a_mw <= _bp_hi)
            _bp_k_mask = (_k_mw >= _bp_lo) & (_k_mw <= _bp_hi)
        else:
            _bp_a_mask = np.ones(len(awake_amps), dtype=bool)
            _bp_k_mask = np.ones(len(keta_amps), dtype=bool)
        n_responsive_awake = _zeta_count(
            rec, unit_ids_unfiltered, awake_onsets[_bp_a_mask], awake_amps[_bp_a_mask]
        )
        n_responsive_keta = _zeta_count(
            rec, unit_ids_unfiltered, keta_onsets[_bp_k_mask], keta_amps[_bp_k_mask]
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

    ac_dir = output_dir / "activation_curve"
    ac_dir.mkdir(exist_ok=True)
    awake_data.amplitude_stats.to_csv(
        ac_dir / "amplitude_response_awake.csv", index=False
    )
    keta_data.amplitude_stats.to_csv(
        ac_dir / f"amplitude_response_{anesthesia_state}.csv", index=False
    )

    find_activation_threshold(
        awake_data.responses_df,
        "awake",
        output_path=ac_dir / "threshold_awake.csv",
    )
    find_activation_threshold(
        keta_data.responses_df,
        anesthesia_state,
        output_path=ac_dir / f"threshold_{anesthesia_state}.csv",
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

    # compute per-cortical-layer PSTHs (no responsiveness filter)
    if LAYER_PSTH_AMPLITUDES is not None:
        lo_mw, hi_mw = LAYER_PSTH_AMPLITUDES
        _a_mw = np.array([_volt_to_mw(v) for v in awake_amps])
        _k_mw = np.array([_volt_to_mw(v) for v in keta_amps])
        layer_awake_onsets = awake_onsets[(_a_mw >= lo_mw) & (_a_mw <= hi_mw)]
        layer_awake_amps = awake_amps[(_a_mw >= lo_mw) & (_a_mw <= hi_mw)]
        layer_keta_onsets = keta_onsets[(_k_mw >= lo_mw) & (_k_mw <= hi_mw)]
        layer_keta_amps = keta_amps[(_k_mw >= lo_mw) & (_k_mw <= hi_mw)]
    else:
        layer_awake_onsets, layer_awake_amps = awake_onsets, awake_amps
        layer_keta_onsets, layer_keta_amps = keta_onsets, keta_amps

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
            layer_awake_onsets,
            layer_awake_amps,
            rec.stateTimes["awake"],
        )
        k_state = process_state(
            rec,
            layer_ids,
            layer_unique,
            layer_keta_onsets,
            layer_keta_amps,
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

    per_amp_psths = _compute_per_amp_psths(
        rec,
        unit_ids,
        unique_ids,
        awake_onsets,
        awake_amps,
        keta_onsets,
        keta_amps,
        raster_uids,
        baseline_stats=None,
        all_unit_ids=list(awake_data_all.psth_unit_ids),
    )

    # per-neuron baseline FR from pre-stimulus PSTH bins (all neurons passing filter)
    bc_all = awake_data_all.bin_centers
    pre_mask = bc_all < 0
    awake_frs = awake_data_all.neuron_psths[:, pre_mask].mean(axis=1).tolist()
    keta_frs = keta_data_all.neuron_psths[:, pre_mask].mean(axis=1).tolist()

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
        n_neurons_total=n_neurons_total,
        n_responsive_awake=n_responsive_awake,
        n_responsive_keta=n_responsive_keta,
        anesthesia_state=anesthesia_state,
        per_amp_psths=per_amp_psths,
        awake_firing_rates=awake_frs,
        keta_firing_rates=keta_frs,
        awake_neuron_thresholds=neuron_thresholds_mw(awake_data.resp_prob_df),
        keta_neuron_thresholds=neuron_thresholds_mw(keta_data.resp_prob_df),
    )


def pool_sessions(results: List[SessionResult]) -> dict:
    """concatenate per-neuron data across sessions and recompute group-level summaries."""
    awake_resp = pd.concat([r.awake.responses_df for r in results], ignore_index=True)
    keta_resp = pd.concat([r.ketamine.responses_df for r in results], ignore_index=True)

    awake_stats = aggregate_by_amplitude(awake_resp)
    keta_stats = aggregate_by_amplitude(keta_resp)

    awake_resp_prob = pd.concat([r.awake.resp_prob_df for r in results], ignore_index=True)
    keta_resp_prob = pd.concat([r.ketamine.resp_prob_df for r in results], ignore_index=True)

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
    all_amps = sorted(
        {amp for r in results if r.per_amp_psths for amp in r.per_amp_psths}
    )
    pooled_per_amp = {}
    for amp in all_amps:

        def _get(r, key, idx):
            return (
                r.per_amp_psths[amp][key][idx]
                if r.per_amp_psths and amp in r.per_amp_psths
                else None
            )

        a_mats = [
            _get(r, "awake", 3) for r in results if _get(r, "awake", 3) is not None
        ]
        k_mats = [_get(r, "keta", 3) for r in results if _get(r, "keta", 3) is not None]
        aa_mats = [
            r.per_amp_psths[amp]["awake_all"][3]
            for r in results
            if r.per_amp_psths
            and amp in r.per_amp_psths
            and "awake_all" in r.per_amp_psths[amp]
        ]
        ak_mats = [
            r.per_amp_psths[amp]["keta_all"][3]
            for r in results
            if r.per_amp_psths
            and amp in r.per_amp_psths
            and "keta_all" in r.per_amp_psths[amp]
        ]
        a_rasters = [
            spike
            for r in results
            if r.per_amp_psths and amp in r.per_amp_psths
            for spike in r.per_amp_psths[amp]["awake"][4]
        ]
        k_rasters = [
            spike
            for r in results
            if r.per_amp_psths and amp in r.per_amp_psths
            for spike in r.per_amp_psths[amp]["keta"][4]
        ]
        if not a_mats:
            continue
        a_all = np.vstack(a_mats)
        k_all = np.vstack(k_mats)
        na, nk = len(a_all), len(k_all)
        entry = {
            "awake": (
                bin_centers,
                np.mean(a_all, axis=0),
                np.std(a_all, axis=0) / np.sqrt(na),
                a_all,
                a_rasters,
            ),
            "keta": (
                bin_centers,
                np.mean(k_all, axis=0),
                np.std(k_all, axis=0) / np.sqrt(nk),
                k_all,
                k_rasters,
            ),
        }
        if aa_mats:
            aa = np.vstack(aa_mats)
            ak = np.vstack(ak_mats)
            entry["awake_all"] = (
                bin_centers,
                np.mean(aa, axis=0),
                np.std(aa, axis=0) / np.sqrt(len(aa)),
                aa,
            )
            entry["keta_all"] = (
                bin_centers,
                np.mean(ak, axis=0),
                np.std(ak, axis=0) / np.sqrt(len(ak)),
                ak,
            )
        entry["n_responsive_awake"] = [
            r.per_amp_psths[amp]["n_responsive_awake"]
            for r in results
            if r.per_amp_psths and amp in r.per_amp_psths
        ]
        entry["n_responsive_keta"] = [
            r.per_amp_psths[amp]["n_responsive_keta"]
            for r in results
            if r.per_amp_psths and amp in r.per_amp_psths
        ]
        pooled_per_amp[amp] = entry

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
        "awake_firing_rates": [
            fr for r in results for fr in (r.awake_firing_rates or [])
        ],
        "keta_firing_rates": [
            fr for r in results for fr in (r.keta_firing_rates or [])
        ],
        "awake_resp_prob": awake_resp_prob,
        "keta_resp_prob": keta_resp_prob,
        "awake_neuron_thresholds": [
            t for r in results for t in (r.awake_neuron_thresholds or [])
        ],
        "keta_neuron_thresholds": [
            t for r in results for t in (r.keta_neuron_thresholds or [])
        ],
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


def _wilcoxon(*args, **kwargs):
    """wilcoxon wrapper that suppresses degenerate-data RuntimeWarnings (e.g. all-zero differences)."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="Degrees of freedom"
        )
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="invalid value"
        )
        return scipy.stats.wilcoxon(*args, **kwargs)


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
        stat, p = _wilcoxon(sub["response_awake"], sub["response_keta"])
        rows.append({"amplitude": amp, "statistic": stat, "p_value": p, "n": len(sub)})

    stats_df = pd.DataFrame(rows)
    stats_df["significant"] = _bh_correction(stats_df["p_value"].values)
    stats_df["n_neurons"] = n_neurons
    stats_df["n_sessions"] = n_sessions
    return stats_df


def _bc_psth(bc: np.ndarray, psth: np.ndarray) -> np.ndarray:
    """subtract pre-stimulus mean from psth (baseline correction)."""
    pre = psth[bc < 0]
    return psth - (pre.mean() if len(pre) else 0.0)


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
        stat, p = _wilcoxon(awake_vals, keta_vals)
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

    if responses_df.empty:
        return {}

    per_neuron = (
        responses_df.groupby(["unique_id", "amplitude"])["response"]
        .mean()
        .reset_index()
    )
    amps = np.array(sorted(per_neuron["amplitude"].unique()))
    if len(amps) == 0:
        return {}

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
        _, p = _wilcoxon(vals, alternative="greater")
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
        sig_mw = np.sort(
            [_volt_to_mw(a) for a in stats_df.loc[stats_df["significant"], "amplitude"]]
        )
        if len(sig_mw):
            all_mw = np.sort(
                np.unique(awake_stats["amplitude"].map(_volt_to_mw).values)
            )
            half_step = np.diff(all_mw).min() / 2 if len(all_mw) > 1 else 1.0
            # group runs of significant intensities that are adjacent in the tested set
            sig_idx = [int(np.argmin(np.abs(all_mw - v))) for v in sig_mw]
            runs, run = [], [sig_idx[0]]
            for pi, ci in zip(sig_idx[:-1], sig_idx[1:]):
                if ci == pi + 1:
                    run.append(ci)
                else:
                    runs.append(run)
                    run = [ci]
            runs.append(run)
            for run in runs:
                ax.hlines(
                    1.04,
                    all_mw[run[0]] - half_step,
                    all_mw[run[-1]] + half_step,
                    colors=_SIG_COLOR,
                    linewidth=2,
                    transform=ax.get_xaxis_transform(),
                    clip_on=True,
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
    if also_save_log:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=6))
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        # refit xlim to data range so sig bar extensions don't go out of bounds
        all_mw_vals = np.sort(
            np.unique(awake_stats["amplitude"].map(_volt_to_mw).values)
        )
        all_mw_vals = all_mw_vals[all_mw_vals > 0]
        if len(all_mw_vals) > 1:
            log_pad = np.exp(np.diff(np.log(all_mw_vals)).min() / 2)
            ax.set_xlim(all_mw_vals[0] / log_pad, all_mw_vals[-1] * log_pad)
        plt.tight_layout()
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
    if not has_raster:
        legend_outside = True
    show_legend = PSTH_SHOW_LEGEND

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
        if len(psth_stats_df) == 1:
            # single-test case (run_psth_stats): show asterisk text above the traces
            row = psth_stats_df.iloc[0]
            p = row["p_value"]
            if p <= 0.001:
                asterisks = "***"
            elif p <= 0.01:
                asterisks = "**"
            elif p <= ALPHA:
                asterisks = "*"
            else:
                asterisks = "ns"
            x_center = (row["bin_start"] + row["bin_end"]) / 2
            has_sig_bars = asterisks != "ns"
            if has_sig_bars:
                ylim_top = bar_y + y_pad * 3
            ax.text(
                x_center,
                bar_y,
                asterisks,
                ha="center",
                va="bottom",
                color=_SIG_COLOR,
                fontsize=14,
                fontweight="bold",
            )
        elif len(sig_bins):
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

    if show_legend:
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
        legend_ax = ax_raster if has_raster else ax
        if legend_outside:
            legend_ax.legend(
                handles=handles,
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                borderaxespad=0,
            )
        else:
            legend_ax.legend(handles=handles, frameon=True, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if normalize:
        yticks = list(range(int(np.floor(data_ymin)), int(np.ceil(data_ymax)) + 1))
    elif ZSCORE:
        top_tick = int(np.ceil(data_ymax / 100) * 100)
        yticks = [0, top_tick // 2, top_tick]
    else:
        top_rounded = int(round(data_ymax))
        if top_rounded == 0 and data_ymax > 0:
            # sub-0.5 Hz — round to 1 sig fig (e.g. 0.3, 0.04)
            from math import floor, log10

            mag = 10 ** floor(log10(data_ymax))
            top_tick = round(data_ymax / mag) * mag
        else:
            top_tick = top_rounded
        yticks = [0, top_tick]
        ylim_bottom = 0
        if not has_sig_bars:
            ylim_top = data_ymax  # actual max including SEM
    ax.set_yticks(yticks)
    if ylim_top <= ylim_bottom:
        ylim_top = ylim_bottom + 1.0
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

    # baseline-correct each neuron by subtracting its pre-stimulus mean (nan-safe)
    pre_mask = bin_centers < 0
    if pre_mask.any() and awake_mat.shape[0] > 0:
        awake_mat = awake_mat - np.nanmean(
            awake_mat[:, pre_mask], axis=1, keepdims=True
        )
    if pre_mask.any() and keta_mat.shape[0] > 0:
        keta_mat = keta_mat - np.nanmean(keta_mat[:, pre_mask], axis=1, keepdims=True)
    if awake_mat.shape == keta_mat.shape:
        diff_mat = keta_mat - awake_mat

    combined = np.concatenate([awake_mat.ravel(), keta_mat.ravel()])
    finite_vals = combined[np.isfinite(combined)]
    if finite_vals.size == 0:
        plt.close("all")
        return
    vmax = np.percentile(finite_vals, HEATMAP_CLIM_PERCENTILE)
    vmin = np.percentile(finite_vals, 100 - HEATMAP_CLIM_PERCENTILE)
    if vmax == vmin:
        plt.close("all")
        return
    diff_finite = diff_mat.ravel() if diff_mat.size > 0 else np.array([])
    diff_finite = diff_finite[np.isfinite(diff_finite)]
    diff_lim = (
        np.percentile(np.abs(diff_finite), HEATMAP_CLIM_PERCENTILE)
        if diff_finite.size > 0
        else 1.0
    )

    ylabel = "Z-score" if ZSCORE else "ΔFR (Hz)"
    diff_ylabel = "ΔZ" if ZSCORE else "ΔFR (Hz)"

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
    awake_counts: Optional[List[int]] = None,
    keta_counts: Optional[List[int]] = None,
):
    """thin boxplot comparing n_responsive_awake vs n_responsive_anesthesia across sessions.

    awake_counts/keta_counts can be passed explicitly for per-amplitude calls;
    otherwise they are read from the SessionResult objects.
    """
    plt.rcParams.update(_ACTIVE_STYLE)

    if awake_counts is None:
        awake_counts = [r.n_responsive_awake for r in results]
    if keta_counts is None:
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

    def _baseline_subtract(psth, sem, bc):
        pre = psth[bc < 0]
        baseline = np.median(pre) if len(pre) else 0.0
        return psth - baseline, sem

    def _smooth(arr):
        if not PSTH_LAYER_SMOOTH_SIGMA_MS:
            return arr
        sigma_bins = PSTH_LAYER_SMOOTH_SIGMA_MS / PSTH_BIN_SIZE
        return scipy.ndimage.gaussian_filter1d(arr, sigma=sigma_bins)

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
    data_min = np.inf
    data_max = -np.inf
    for i, lyr in enumerate(present):
        offset = i * PSTH_LAYER_OFFSET
        bc, psth, sem = layer_psths[lyr][state][:3]
        psth, sem = _baseline_subtract(psth, sem, bc)
        psth, sem = _smooth(psth), _smooth(sem)
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
        onset_idx = int(np.argmin(np.abs(bc)))
        data_min = min(data_min, (psth + offset)[onset_idx])
        data_max = max(data_max, (psth + offset).max())

    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_max <= data_min:
        ylim_lo, ylim_hi = ytick_pos[0] - 0.15, ytick_pos[-1] + 1.1
    else:
        y_span = data_max - data_min
        ylim_lo = data_min - 0.05 * y_span
        ylim_hi = data_max

    xmin, xmax = -PSTH_WINDOW[0], PSTH_WINDOW[1]
    ax.set_xlim(xmin, xmax)
    ax.set_xticks([0, int(xmax // 2), int(xmax)])
    ax.set_xticklabels(["0", str(int(xmax // 2)), str(int(xmax))])
    ax.set_xlabel("Time from Onset (ms)")
    # ax.set_ylabel("Layer")
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([f"L{lyr}" for lyr in present])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(ylim_lo, ylim_hi)

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

    # baseline-correct per neuron (subtract pre-stim mean, nan-safe)
    pre_mask_resp = bc < 0
    if pre_mask_resp.any():
        awake_mat = awake_mat - np.nanmean(
            awake_mat[:, pre_mask_resp], axis=1, keepdims=True
        )
        keta_mat = keta_mat - np.nanmean(
            keta_mat[:, pre_mask_resp], axis=1, keepdims=True
        )

    # sort by mean awake firing rate in PULSE_WINDOW (descending = highest at top)
    win_mask = (bc >= PULSE_WINDOW[0]) & (bc <= PULSE_WINDOW[1])
    sort_key = (
        awake_mat[:, win_mask].mean(axis=1)
        if win_mask.any()
        else awake_mat.mean(axis=1)
    )
    order = np.argsort(sort_key)[::-1]
    awake_mat = awake_mat[order]
    keta_mat = keta_mat[order]

    N, n_bins = awake_mat.shape
    combined = np.concatenate([awake_mat, keta_mat], axis=1)  # (N, 2*n_bins)

    finite_vals = combined.ravel()[np.isfinite(combined.ravel())]
    if finite_vals.size == 0:
        return
    vmax = np.percentile(finite_vals, HEATMAP_CLIM_PERCENTILE)
    vmin = np.percentile(finite_vals, 100 - HEATMAP_CLIM_PERCENTILE)
    if vmax == vmin:
        return

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
    cb.set_label("ΔFR (Hz)")
    if DARK_MODE:
        cb.ax.yaxis.set_tick_params(color="white")
        cb.outline.set_edgecolor("white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    _save(output_path, dpi=300)
    plt.close()


def plot_firing_rate_comparison(
    awake_frs: List[float],
    keta_frs: List[float],
    output_path: Path,
    keta_color: str = ANESTHESIA_COLORS["ketamine"],
    keta_label: str = "Ketamine",
):
    """boxplot comparing per-neuron overall firing rates between awake and anesthesia."""
    plt.rcParams.update(_ACTIVE_STYLE)
    fig_w = _PSTH_M + 1 + _PSTH_M
    fig_h = _PSTH_M + PSTH_AX_H + _PSTH_M
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([_PSTH_M / fig_w, _PSTH_M / fig_h, 1 / fig_w, PSTH_AX_H / fig_h])

    bp = ax.boxplot(
        [awake_frs, keta_frs],
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
    for x_center, frs, color in [
        (1, awake_frs, COLOR_AWAKE),
        (2, keta_frs, keta_color),
    ]:
        jitter = rng.uniform(-0.1, 0.1, len(frs))
        ax.scatter(x_center + jitter, frs, color=color, s=20, zorder=5, alpha=0.1)

    from scipy.stats import mannwhitneyu

    _, p = mannwhitneyu(awake_frs, keta_frs, alternative="two-sided")
    sig_label = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    whisker_vals = [w.get_ydata()[1] for w in bp["whiskers"]]
    y_lo = min(whisker_vals)
    y_hi = max(whisker_vals)
    span = max(y_hi - y_lo, 0.1)
    ax.set_ylim(y_lo - 0.1 * span, y_hi + 0.3 * span)

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
    ax.set_ylabel("Firing Rate (Hz)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    _save(output_path, dpi=300)
    plt.close()


def plot_resp_prob_curve(
    awake_prob_df: pd.DataFrame,
    keta_prob_df: pd.DataFrame,
    output_path: Path,
    title: str = "Response Probability",
    keta_color: str = ANESTHESIA_COLORS["ketamine"],
    keta_label: str = "Ketamine",
):
    """activation curve with y = mean response probability across neurons (0–1)."""
    plt.rcParams.update(_ACTIVE_STYLE)
    fig, ax = plt.subplots()

    for df, color, label in [
        (awake_prob_df, COLOR_AWAKE, "Awake"),
        (keta_prob_df, keta_color, keta_label),
    ]:
        stats = (
            df.groupby("amplitude")["response_prob"]
            .agg(mean="mean", sem=lambda x: x.std() / np.sqrt(len(x)))
            .reset_index()
        )
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

    all_mw = pd.concat(
        [awake_prob_df["amplitude"], keta_prob_df["amplitude"]]
    ).map(_volt_to_mw)
    ax.set_xlim(all_mw.min(), all_mw.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Illumination Intensity (mW/mm²)")
    ax.set_ylabel(f"Response Probability (k={RESP_PROB_K})")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(output_path, dpi=300)
    plt.close()


def plot_resp_prob_thresholds(
    awake_thresholds: List[float],
    keta_thresholds: List[float],
    output_path: Path,
    keta_color: str = ANESTHESIA_COLORS["ketamine"],
    keta_label: str = "Ketamine",
):
    """boxplot + jitter of per-neuron response-probability thresholds by state."""
    plt.rcParams.update(_ACTIVE_STYLE)
    fig_w = _PSTH_M + 1.5 + _PSTH_M
    fig_h = _PSTH_M + PSTH_AX_H + _PSTH_M
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([_PSTH_M / fig_w, _PSTH_M / fig_h, 1.5 / fig_w, PSTH_AX_H / fig_h])

    data = [awake_thresholds, keta_thresholds]
    colors = [COLOR_AWAKE, keta_color]
    labels = ["Awake", keta_label]

    bp = ax.boxplot(
        data,
        positions=[1, 2],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=_FOREGROUND_COLOR, linewidth=2),
        whiskerprops=dict(color=_FOREGROUND_COLOR, linewidth=1.5),
        capprops=dict(color=_FOREGROUND_COLOR, linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_linewidth(0)

    rng = np.random.default_rng(0)
    for x_center, vals, color in zip([1, 2], data, colors):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            x_center + jitter, vals, color=color, alpha=0.6, s=18, linewidths=0, zorder=3
        )

    # Wilcoxon test if both have data
    if len(awake_thresholds) >= 5 and len(keta_thresholds) >= 5:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(awake_thresholds, keta_thresholds, alternative="two-sided")
        y_top = max(max(awake_thresholds, default=0), max(keta_thresholds, default=0))
        ax.plot([1, 2], [y_top * 1.07, y_top * 1.07], color=_FOREGROUND_COLOR, linewidth=1)
        pstr = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.text(1.5, y_top * 1.09, pstr, ha="center", va="bottom", fontsize=10)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Response Threshold (mW/mm²)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(output_path, dpi=300)
    plt.close()


def plot_responsive_count_curve(
    per_amp_psths: dict,
    output_path: Path,
    stats_df: Optional[pd.DataFrame] = None,
    keta_color: str = ANESTHESIA_COLORS["ketamine"],
    keta_label: str = "Ketamine",
    also_save_log: bool = False,
):
    """activation-curve-style plot of responsive neuron counts vs. stimulus intensity."""
    plt.rcParams.update(_ACTIVE_STYLE)
    amps = sorted(per_amp_psths.keys())
    mw_vals = np.array([_volt_to_mw(a) for a in amps])

    # sum per-session counts across sessions for each amplitude
    a_sum = np.array(
        [sum(per_amp_psths[a]["n_responsive_awake"]) for a in amps], dtype=float
    )
    k_sum = np.array(
        [sum(per_amp_psths[a]["n_responsive_keta"]) for a in amps], dtype=float
    )

    fig, ax = plt.subplots()
    ax.plot(mw_vals, a_sum, color=COLOR_AWAKE, marker="o", label="Awake")
    ax.plot(mw_vals, k_sum, color=keta_color, marker="o", label=keta_label)

    if stats_df is not None:
        sig_mw = np.sort(
            [_volt_to_mw(a) for a in stats_df.loc[stats_df["significant"], "amplitude"]]
        )
        if len(sig_mw):
            all_mw = np.sort(np.unique(mw_vals))
            half_step = np.diff(all_mw).min() / 2 if len(all_mw) > 1 else 1.0
            sig_idx = [int(np.argmin(np.abs(all_mw - v))) for v in sig_mw]
            runs, run = [], [sig_idx[0]]
            for pi, ci in zip(sig_idx[:-1], sig_idx[1:]):
                if ci == pi + 1:
                    run.append(ci)
                else:
                    runs.append(run)
                    run = [ci]
            runs.append(run)
            for run in runs:
                ax.hlines(
                    1.04,
                    all_mw[run[0]] - half_step,
                    all_mw[run[-1]] + half_step,
                    colors=_SIG_COLOR,
                    linewidth=2,
                    transform=ax.get_xaxis_transform(),
                    clip_on=True,
                )

    ax.set_xlim(mw_vals.min(), mw_vals.max())
    ax.set_xlabel("Illumination Intensity (mW/mm²)")
    ax.set_ylabel("No. responsive\nneurons")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    top = max(a_sum.max(), k_sum.max())
    ax.set_ylim(bottom=0)
    ax.set_yticks([0, int(round(top))])
    plt.tight_layout()
    _save(output_path, dpi=300)
    if also_save_log:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=6))
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        pos_mw = mw_vals[mw_vals > 0]
        if len(pos_mw) > 1:
            log_pad = np.exp(np.diff(np.log(pos_mw)).min() / 2)
            ax.set_xlim(pos_mw.min() / log_pad, pos_mw.max() * log_pad)
        plt.tight_layout()
        _save(
            output_path.with_name(output_path.stem + "_log" + output_path.suffix),
            dpi=300,
        )
    plt.close()


def plot_session(result: SessionResult):
    """save per-session activation curve, PSTH, and PSTH heatmap."""
    ac = result.anesthesia_state
    ac_color = ANESTHESIA_COLORS.get(ac, ANESTHESIA_COLORS["ketamine"])
    ac_label = ac.title()
    od = result.output_dir

    # create subfolders
    ac_dir = od / "activation_curve"
    pr_dir = od / "psth_responsive"
    pa_dir = od / "psth_all"
    pl_dir = od / "psth_layers"
    ph_dir = od / "psth_heatmap"
    nr_dir = od / "n_responsive"
    bc_dir = pr_dir / "baseline_corrected"
    for d in (ac_dir, pr_dir, pa_dir, pl_dir, ph_dir, nr_dir, bc_dir):
        d.mkdir(exist_ok=True)

    plot_activation_curve(
        result.awake.amplitude_stats,
        result.ketamine.amplitude_stats,
        ac_dir / "activation_curve.pdf",
        title=result.session_name,
        keta_color=ac_color,
        keta_label=ac_label,
    )
    plot_resp_prob_curve(
        result.awake.resp_prob_df,
        result.ketamine.resp_prob_df,
        ac_dir / "response_probability_curve.pdf",
        title=result.session_name,
        keta_color=ac_color,
        keta_label=ac_label,
    )

    _a_bc = result.awake.bin_centers
    _k_bc = result.ketamine.bin_centers
    plot_psth(
        (_a_bc, result.awake.psth, result.awake.psth_sem),
        (_k_bc, result.ketamine.psth, result.ketamine.psth_sem),
        pr_dir / "psth_responsive.pdf",
        title=result.session_name,
        awake_raster=result.awake.raster_spikes,
        keta_raster=result.ketamine.raster_spikes,
        window=PSTH_WINDOW_RESPONSIVE,
        keta_color=ac_color,
        keta_label=ac_label,
    )
    plot_psth(
        (_a_bc, _bc_psth(_a_bc, result.awake.psth), result.awake.psth_sem),
        (_k_bc, _bc_psth(_k_bc, result.ketamine.psth), result.ketamine.psth_sem),
        bc_dir / "psth_responsive.pdf",
        title=result.session_name,
        awake_raster=result.awake.raster_spikes,
        keta_raster=result.ketamine.raster_spikes,
        window=PSTH_WINDOW_RESPONSIVE,
        keta_color=ac_color,
        keta_label=ac_label,
    )

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
        pa_dir / "psth_all_cortex.pdf",
        **_psth_all_kwargs,
    )
    for area in ("Th", "Ca1"):
        if result.area_psths and area in result.area_psths:
            plot_psth(
                _awake_all_data,
                _keta_all_data,
                pa_dir / f"psth_all_{area}.pdf",
                extra_traces=_build_extra_traces(result.area_psths, [area]),
                **_psth_all_kwargs,
            )

    _layer_amp_tag = (
        f"_{int(LAYER_PSTH_AMPLITUDES[0])}_{int(LAYER_PSTH_AMPLITUDES[1])}mW"
        if LAYER_PSTH_AMPLITUDES is not None
        else ""
    )
    plot_psth_layers(
        result.layer_psths,
        pl_dir / f"psth_layers_awake{_layer_amp_tag}.pdf",
        state="awake",
    )
    plot_psth_layers(
        result.layer_psths,
        pl_dir / f"psth_layers_{ac}{_layer_amp_tag}.pdf",
        state="keta",
    )

    plot_psth_heatmap(result, ph_dir / "psth_heatmap.pdf")

    # per-amplitude plots
    if result.per_amp_psths:
        for amp, data in result.per_amp_psths.items():
            mw = _volt_to_mw(amp)
            mw_str = f"{mw:.1f}".rstrip("0").rstrip(".")
            amp_tag = mw_str
            a_bc, a_psth, a_sem, a_mats, a_raster = data["awake"]
            k_bc, k_psth, k_sem, k_mats, k_raster = data["keta"]
            amp_stats_df = run_psth_stats(a_mats, k_mats, a_bc, len(a_mats))
            plot_psth(
                (a_bc, a_psth, a_sem),
                (k_bc, k_psth, k_sem),
                pr_dir / f"psth_responsive_{amp_tag}.pdf",
                title=f"{result.session_name} — {mw_str} mW/mm²",
                psth_stats_df=amp_stats_df,
                awake_raster=a_raster,
                keta_raster=k_raster,
                window=PSTH_WINDOW_RESPONSIVE,
                keta_color=ac_color,
                keta_label=ac_label,
            )
            plot_psth(
                (a_bc, _bc_psth(a_bc, a_psth), a_sem),
                (k_bc, _bc_psth(k_bc, k_psth), k_sem),
                bc_dir / f"psth_responsive_{amp_tag}.pdf",
                title=f"{result.session_name} — {mw_str} mW/mm²",
                psth_stats_df=amp_stats_df,
                awake_raster=a_raster,
                keta_raster=k_raster,
                window=PSTH_WINDOW_RESPONSIVE,
                keta_color=ac_color,
                keta_label=ac_label,
            )
            plot_responsive_counts(
                [],
                nr_dir / f"responsive_counts_{amp_tag}.pdf",
                keta_color=ac_color,
                keta_label=ac_label,
                awake_counts=[data["n_responsive_awake"]],
                keta_counts=[data["n_responsive_keta"]],
            )
            if "awake_all" in data:
                aa_bc, aa_psth, aa_sem, _ = data["awake_all"]
                ak_bc, ak_psth, ak_sem, _ = data["keta_all"]
                plot_psth(
                    (aa_bc, aa_psth, aa_sem),
                    (ak_bc, ak_psth, ak_sem),
                    pa_dir / f"psth_all_cortex_{amp_tag}.pdf",
                    title=f"{result.session_name} — {mw_str} mW/mm² (all)",
                    normalize=True,
                    keta_color=ac_color,
                    keta_label=ac_label,
                )

        # pooled-amplitude PSTH for this session
        if POOLED_AMPLITUDES is not None:
            _pa_lo, _pa_hi = POOLED_AMPLITUDES
            sel_amps = sorted(
                [a for a in result.per_amp_psths if _pa_lo <= _volt_to_mw(a) <= _pa_hi]
            )
            if sel_amps:
                pa_mats = np.vstack(
                    [result.per_amp_psths[a]["awake"][3] for a in sel_amps]
                )
                pk_mats = np.vstack(
                    [result.per_amp_psths[a]["keta"][3] for a in sel_amps]
                )
                pa_raster = [
                    s for a in sel_amps for s in result.per_amp_psths[a]["awake"][4]
                ]
                pk_raster = [
                    s for a in sel_amps for s in result.per_amp_psths[a]["keta"][4]
                ]
                bc = result.per_amp_psths[sel_amps[0]]["awake"][0]
                n_pool = len(pa_mats)
                pool_stats_df = run_psth_stats(pa_mats, pk_mats, bc, n_pool)
                pool_label = "+".join(
                    f"{_volt_to_mw(a):.1f}".rstrip("0").rstrip(".") for a in sel_amps
                )
                _pa_psth = np.mean(pa_mats, axis=0)
                _pk_psth = np.mean(pk_mats, axis=0)
                _pa_sem = np.std(pa_mats, axis=0) / np.sqrt(n_pool)
                _pk_sem = np.std(pk_mats, axis=0) / np.sqrt(n_pool)
                plot_psth(
                    (bc, _pa_psth, _pa_sem),
                    (bc, _pk_psth, _pk_sem),
                    pr_dir / f"psth_responsive_combined_{pool_label}.pdf",
                    title=f"{result.session_name} — {pool_label} mW/mm² ({n_pool} neurons)",
                    psth_stats_df=pool_stats_df,
                    awake_raster=pa_raster,
                    keta_raster=pk_raster,
                    window=PSTH_WINDOW_RESPONSIVE,
                    keta_color=ac_color,
                    keta_label=ac_label,
                )
                plot_psth(
                    (bc, _bc_psth(bc, _pa_psth), _pa_sem),
                    (bc, _bc_psth(bc, _pk_psth), _pk_sem),
                    bc_dir / f"psth_responsive_combined_{pool_label}.pdf",
                    title=f"{result.session_name} — {pool_label} mW/mm² ({n_pool} neurons)",
                    psth_stats_df=pool_stats_df,
                    awake_raster=pa_raster,
                    keta_raster=pk_raster,
                    window=PSTH_WINDOW_RESPONSIVE,
                    keta_color=ac_color,
                    keta_label=ac_label,
                )


def _process_and_plot_session(args: tuple) -> "SessionResult":
    """top-level picklable worker: process one session and save its plots."""
    session_dir, config = args
    result = process_session(Path(session_dir), config)
    plot_session(result)
    return result


def main():
    config = load_config()
    with open("sessions.toml", "rb") as f:
        sessions_cfg = tomllib.load(f)

    dirs = sessions_cfg["sessions"]["dirs"]
    n_workers = min(
        len(dirs), N_WORKERS if N_WORKERS is not None else _available_cores()
    )
    print(
        f"{_TEAL}Parallelism: {n_workers} session worker(s) "
        f"(ZETA sequential per session){_RESET}"
    )

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(
                ex.map(_process_and_plot_session, [(d, config) for d in dirs])
            )
    else:
        results = [_process_and_plot_session((d, config)) for d in dirs]

    print(f"\n{_TEAL}{'─' * 50}")
    for r in results:
        print(f"  {r.session_name}: {len(r.unit_ids)}/{r.n_neurons_total} responsive")
    print(f"{'─' * 50}{_RESET}\n")

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

        # create subfolders
        p_ac_dir = pooled_dir / "activation_curve"
        p_pr_dir = pooled_dir / "psth_responsive"
        p_pa_dir = pooled_dir / "psth_all"
        p_pl_dir = pooled_dir / "psth_layers"
        p_ph_dir = pooled_dir / "psth_heatmap"
        p_nr_dir = pooled_dir / "n_responsive"
        p_cal_dir = pooled_dir / "calibration"
        p_bc_dir = p_pr_dir / "baseline_corrected"
        for d in (
            p_ac_dir,
            p_pr_dir,
            p_pa_dir,
            p_pl_dir,
            p_ph_dir,
            p_nr_dir,
            p_cal_dir,
            p_bc_dir,
        ):
            d.mkdir(exist_ok=True)

        plot_responsive_counts(
            group,
            p_nr_dir / "responsive_counts.pdf",
            keta_color=ac_color,
            keta_label=ac_label,
        )

        plot_calibration(p_cal_dir / "calibration_fit.pdf")

        pooled = pool_sessions(group)
        stats_df = run_stats(
            pooled["awake_resp"],
            pooled["keta_resp"],
            pooled["n_neurons"],
            pooled["n_sessions"],
        )
        stats_df.to_csv(p_ac_dir / "activation_curve_stats.csv", index=False)

        psth_stats_df = run_psth_stats(
            pooled["awake_psths"],
            pooled["keta_psths"],
            pooled["bin_centers"],
            pooled["n_neurons"],
        )
        psth_stats_df.to_csv(p_pr_dir / "psth_stats.csv", index=False)

        find_activation_threshold(
            pooled["awake_resp"],
            "awake",
            output_path=p_ac_dir / "threshold_awake.csv",
        )
        find_activation_threshold(
            pooled["keta_resp"],
            anesthetic,
            output_path=p_ac_dir / f"threshold_{anesthetic}.csv",
        )

        pooled_title = f"Pooled {ac_label} ({pooled['n_neurons']} neurons, {pooled['n_sessions']} sessions)"
        plot_activation_curve(
            pooled["awake_stats"],
            pooled["keta_stats"],
            p_ac_dir / "activation_curve_pooled.pdf",
            stats_df=stats_df,
            title=pooled_title,
            keta_color=ac_color,
            keta_label=ac_label,
            also_save_log=True,
        )
        plot_activation_curve(
            pooled["awake_stats"],
            None,
            p_ac_dir / "activation_curve_pooled_awakeOnly.pdf",
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

        _p_a_bc = pooled["bin_centers"]
        _p_a_psth = pooled["awake_psth"]
        _p_k_psth = pooled["keta_psth"]
        _p_a_sem = pooled["awake_psth_sem"]
        _p_k_sem = pooled["keta_psth_sem"]
        plot_psth(
            (_p_a_bc, _p_a_psth, _p_a_sem),
            (_p_a_bc, _p_k_psth, _p_k_sem),
            p_pr_dir / "psth_pooled_responsive.pdf",
            title=f"Pooled PSTH ({pooled['n_neurons']} responsive neurons)",
            psth_stats_df=psth_stats_df,
            awake_raster=pooled_awake_raster,
            keta_raster=pooled_keta_raster,
            window=PSTH_WINDOW_RESPONSIVE,
            keta_color=ac_color,
            keta_label=ac_label,
        )
        plot_psth(
            (_p_a_bc, _bc_psth(_p_a_bc, _p_a_psth), _p_a_sem),
            (_p_a_bc, _bc_psth(_p_a_bc, _p_k_psth), _p_k_sem),
            p_bc_dir / "psth_pooled_responsive.pdf",
            title=f"Pooled PSTH ({pooled['n_neurons']} responsive neurons)",
            psth_stats_df=psth_stats_df,
            awake_raster=pooled_awake_raster,
            keta_raster=pooled_keta_raster,
            window=PSTH_WINDOW_RESPONSIVE,
            keta_color=ac_color,
            keta_label=ac_label,
        )

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
            p_pa_dir / "psth_pooled_all_cortex.pdf",
            **_psth_all_kwargs,
        )
        for area in ("Th", "Ca1"):
            if pooled.get("area_psths") and area in pooled["area_psths"]:
                plot_psth(
                    _awake_all_data,
                    _keta_all_data,
                    p_pa_dir / f"psth_pooled_all_{area}.pdf",
                    extra_traces=_build_extra_traces(pooled["area_psths"], [area]),
                    **_psth_all_kwargs,
                )

        _layer_amp_tag = (
            f"_{int(LAYER_PSTH_AMPLITUDES[0])}_{int(LAYER_PSTH_AMPLITUDES[1])}mW"
            if LAYER_PSTH_AMPLITUDES is not None
            else ""
        )
        plot_psth_layers(
            pooled.get("layer_psths"),
            p_pl_dir / f"psth_pooled_layers_awake{_layer_amp_tag}.pdf",
            state="awake",
        )
        plot_psth_layers(
            pooled.get("layer_psths"),
            p_pl_dir / f"psth_pooled_layers_{anesthetic}{_layer_amp_tag}.pdf",
            state="keta",
        )

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
            p_ph_dir / "psth_heatmap_pooled.pdf",
            anesthesia_label=ac_label,
        )
        plot_psth_heatmap_responsive_comparison(
            group,
            p_ph_dir / "psth_heatmap_responsive_comparison.pdf",
            anesthesia_label=ac_label,
        )

        # pooled per-amplitude plots
        for amp, data in pooled.get("per_amp_psths", {}).items():
            mw = _volt_to_mw(amp)
            mw_str = f"{mw:.1f}".rstrip("0").rstrip(".")
            amp_tag = mw_str
            a_bc, a_psth, a_sem, a_mats, a_raster = data["awake"]
            k_bc, k_psth, k_sem, k_mats, k_raster = data["keta"]
            amp_stats_df = run_psth_stats(a_mats, k_mats, a_bc, len(a_mats))
            n_resp = len(a_mats)
            plot_psth(
                (a_bc, a_psth, a_sem),
                (k_bc, k_psth, k_sem),
                p_pr_dir / f"psth_pooled_responsive_{amp_tag}.pdf",
                title=f"Pooled PSTH — {mw_str} mW/mm² ({n_resp} neurons)",
                psth_stats_df=amp_stats_df,
                awake_raster=a_raster,
                keta_raster=k_raster,
                window=PSTH_WINDOW_RESPONSIVE,
                keta_color=ac_color,
                keta_label=ac_label,
            )
            plot_psth(
                (a_bc, _bc_psth(a_bc, a_psth), a_sem),
                (k_bc, _bc_psth(k_bc, k_psth), k_sem),
                p_bc_dir / f"psth_pooled_responsive_{amp_tag}.pdf",
                title=f"Pooled PSTH — {mw_str} mW/mm² ({n_resp} neurons)",
                psth_stats_df=amp_stats_df,
                awake_raster=a_raster,
                keta_raster=k_raster,
                window=PSTH_WINDOW_RESPONSIVE,
                keta_color=ac_color,
                keta_label=ac_label,
            )
            plot_responsive_counts(
                [],
                p_nr_dir / f"responsive_counts_{amp_tag}.pdf",
                keta_color=ac_color,
                keta_label=ac_label,
                awake_counts=data.get("n_responsive_awake", []),
                keta_counts=data.get("n_responsive_keta", []),
            )
            if "awake_all" in data:
                aa_bc, aa_psth, aa_sem, _ = data["awake_all"]
                ak_bc, ak_psth, ak_sem, _ = data["keta_all"]
                plot_psth(
                    (aa_bc, aa_psth, aa_sem),
                    (ak_bc, ak_psth, ak_sem),
                    p_pa_dir / f"psth_pooled_all_cortex_{amp_tag}.pdf",
                    title=f"Pooled PSTH all — {mw_str} mW/mm² ({pooled['n_neurons_all']} neurons)",
                    normalize=True,
                    keta_color=ac_color,
                    keta_label=ac_label,
                )

        # pooled-amplitude PSTH: combine trials from POOLED_AMPLITUDES across sessions
        if POOLED_AMPLITUDES is not None:
            per_amp_data = pooled.get("per_amp_psths", {})
            _pa_lo, _pa_hi = POOLED_AMPLITUDES
            sel_amps = sorted(
                [a for a in per_amp_data if _pa_lo <= _volt_to_mw(a) <= _pa_hi]
            )
            if sel_amps:
                pa_mats = np.vstack([per_amp_data[a]["awake"][3] for a in sel_amps])
                pk_mats = np.vstack([per_amp_data[a]["keta"][3] for a in sel_amps])
                pa_raster = [s for a in sel_amps for s in per_amp_data[a]["awake"][4]]
                pk_raster = [s for a in sel_amps for s in per_amp_data[a]["keta"][4]]
                bc = per_amp_data[sel_amps[0]]["awake"][0]
                n_pool = len(pa_mats)
                pool_stats_df = run_psth_stats(pa_mats, pk_mats, bc, n_pool)
                pool_label = "+".join(
                    f"{_volt_to_mw(a):.1f}".rstrip("0").rstrip(".") for a in sel_amps
                )
                _pp_a_psth = np.mean(pa_mats, axis=0)
                _pp_k_psth = np.mean(pk_mats, axis=0)
                _pp_a_sem = np.std(pa_mats, axis=0) / np.sqrt(n_pool)
                _pp_k_sem = np.std(pk_mats, axis=0) / np.sqrt(n_pool)
                plot_psth(
                    (bc, _pp_a_psth, _pp_a_sem),
                    (bc, _pp_k_psth, _pp_k_sem),
                    p_pr_dir / f"psth_pooled_responsive_combined_{pool_label}.pdf",
                    title=f"Pooled PSTH — {pool_label} mW/mm² ({n_pool} neurons)",
                    psth_stats_df=pool_stats_df,
                    awake_raster=pa_raster,
                    keta_raster=pk_raster,
                    window=PSTH_WINDOW_RESPONSIVE,
                    keta_color=ac_color,
                    keta_label=ac_label,
                )
                plot_psth(
                    (bc, _bc_psth(bc, _pp_a_psth), _pp_a_sem),
                    (bc, _bc_psth(bc, _pp_k_psth), _pp_k_sem),
                    p_bc_dir / f"psth_pooled_responsive_combined_{pool_label}.pdf",
                    title=f"Pooled PSTH — {pool_label} mW/mm² ({n_pool} neurons)",
                    psth_stats_df=pool_stats_df,
                    awake_raster=pa_raster,
                    keta_raster=pk_raster,
                    window=PSTH_WINDOW_RESPONSIVE,
                    keta_color=ac_color,
                    keta_label=ac_label,
                )

        # firing rate comparison boxplot
        plot_firing_rate_comparison(
            pooled["awake_firing_rates"],
            pooled["keta_firing_rates"],
            p_ac_dir / "baseline_FRs.pdf",
            keta_color=ac_color,
            keta_label=ac_label,
        )

        # response probability curve and per-neuron threshold boxplot
        plot_resp_prob_curve(
            pooled["awake_resp_prob"],
            pooled["keta_resp_prob"],
            p_ac_dir / "response_probability_curve.pdf",
            title=pooled_title,
            keta_color=ac_color,
            keta_label=ac_label,
        )
        plot_resp_prob_thresholds(
            pooled["awake_neuron_thresholds"],
            pooled["keta_neuron_thresholds"],
            p_ac_dir / "response_probability_thresholds.pdf",
            keta_color=ac_color,
            keta_label=ac_label,
        )

        # responsive neuron count curve
        if pooled.get("per_amp_psths"):
            plot_responsive_count_curve(
                pooled["per_amp_psths"],
                p_nr_dir / "activation_curve_responsive_counts.pdf",
                stats_df=stats_df,
                keta_color=ac_color,
                keta_label=ac_label,
                also_save_log=True,
            )

        # baseline-subtracted activation curve
        bc_all = group[0].awake_all.bin_centers
        pre_mask_all = bc_all < 0
        a_bl_map, k_bl_map = {}, {}
        for r in group:
            all_uid_to_row = {u: i for i, u in enumerate(r.awake_all.psth_unit_ids)}
            for uid in r.awake.psth_unit_ids:
                if uid in all_uid_to_row:
                    a_bl_map[f"{r.session_name}_{uid}"] = r.awake_all.neuron_psths[
                        all_uid_to_row[uid], pre_mask_all
                    ].mean()
            all_uid_to_row_k = {
                u: i for i, u in enumerate(r.ketamine_all.psth_unit_ids)
            }
            for uid in r.ketamine.psth_unit_ids:
                if uid in all_uid_to_row_k:
                    k_bl_map[f"{r.session_name}_{uid}"] = r.ketamine_all.neuron_psths[
                        all_uid_to_row_k[uid], pre_mask_all
                    ].mean()
        awake_resp_bl = pooled["awake_resp"].copy()
        keta_resp_bl = pooled["keta_resp"].copy()
        awake_resp_bl["response"] -= awake_resp_bl["unique_id"].map(a_bl_map).fillna(0)
        keta_resp_bl["response"] -= keta_resp_bl["unique_id"].map(k_bl_map).fillna(0)
        awake_stats_bl = aggregate_by_amplitude(awake_resp_bl)
        keta_stats_bl = aggregate_by_amplitude(keta_resp_bl)
        plot_activation_curve(
            awake_stats_bl,
            keta_stats_bl,
            p_ac_dir / "activation_curve_pooled_baseline_subtracted.pdf",
            stats_df=stats_df,
            title=pooled_title + " (baseline subtracted)",
            keta_color=ac_color,
            keta_label=ac_label,
            also_save_log=True,
        )


if __name__ == "__main__":
    main()
