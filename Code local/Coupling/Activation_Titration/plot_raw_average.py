"""stimulus-triggered average of raw neuropixels AP band data."""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomllib
from scipy.ndimage import gaussian_filter

SESSIONS_TOML = Path("sessions.toml")
OUTPUT_DIR_NAME = "output"

WINDOW_MS = (-10.0, 20.0)  # (pre, post) ms relative to onset
STIM_AMPLITUDE_RANGE = (0.5, 2)  # (min_V, max_V) inclusive; None = no limit
SMOOTH_SIGMA = None  # gaussian smoothing (channels × samples)
CLIM_PERCENTILE = 80  # percentile for symmetric colour limits
DEPTH_RANGE_UM = (
    None,
    1200,
)  # (min_um, max_um) or None for all channels


def _read_meta(bin_path: Path) -> dict:
    meta = {}
    with bin_path.with_suffix(".meta").open() as f:
        for line in f.read().splitlines():
            k, v = line.split("=", 1)
            meta[k.lstrip("~")] = v
    return meta


def _samp_rate(meta: dict) -> float:
    return float(meta["imSampRate"])


def _int2volts(meta: dict) -> float:
    return float(meta["imAiRangeMax"]) / int(meta.get("imMaxInt", 512))


def _ap_gains(meta: dict) -> np.ndarray:
    np1_types = {0, 1020, 1030, 1200, 1100, 1120, 1121, 1122, 1123, 1300}
    n_ap = int(meta["acqApLfSy"].split(",")[0])
    gains = np.zeros(n_ap)
    probe_type = int(meta.get("imDatPrb_type", 0))
    if probe_type in np1_types:
        entries = meta["imroTbl"].split(")")
        for i in range(n_ap):
            gains[i] = float(entries[i + 1].split()[3])
    elif "imChan0apGain" in meta:
        gains[:] = float(meta["imChan0apGain"])
    elif probe_type in {21, 24}:
        gains[:] = 80
    elif probe_type == 2013:
        gains[:] = 100
    else:
        gains[:] = 1
    return gains


def _chan_counts_im(meta: dict) -> Tuple[int, int, int]:
    ap, lf, sy = meta["snsApLfSy"].split(",")
    return int(ap), int(lf), int(sy)


def _make_memmap(bin_path: Path, meta: dict) -> np.ndarray:
    n_chan = int(meta["nSavedChans"])
    n_samp = int(meta["fileSizeBytes"]) // (2 * n_chan)
    return np.memmap(
        bin_path, dtype="int16", mode="r", shape=(n_chan, n_samp), offset=0, order="F"
    )


def extract_voltage_snapshot(
    raw_data: np.ndarray,
    trigger_samples: np.ndarray,
    channels: np.ndarray,
    window_ms: Tuple[float, float],
    srate: float,
    fI2V: float,
    gains: np.ndarray,
    smooth_sigma: float = 2.0,
) -> Tuple[np.ndarray, int]:
    """average raw AP data around triggers, baseline-correct per snippet, convert to µV."""
    bins = np.arange(
        int(window_ms[0] * 1e-3 * srate),
        int(window_ms[1] * 1e-3 * srate),
    )
    n_pre = int(-window_ms[0] * 1e-3 * srate)  # samples before stim onset
    n_file_samp = raw_data.shape[1]
    snapshot = np.zeros((len(channels), len(bins)), dtype=np.float64)
    n_valid = 0
    for t in trigger_samples:
        b = t + bins
        if b[0] < 0 or b[-1] >= n_file_samp:
            continue
        snippet = raw_data[np.ix_(channels, b)].astype(np.float64)
        snippet -= snippet[:, :n_pre].mean(axis=1, keepdims=True)
        snapshot += snippet
        n_valid += 1
    if n_valid == 0:
        raise ValueError("No valid triggers within file bounds.")
    snapshot /= n_valid
    snapshot *= ((fI2V / gains[channels]) * 1e6)[:, np.newaxis]
    if smooth_sigma is not None and smooth_sigma > 0:
        snapshot = gaussian_filter(snapshot, smooth_sigma)
    return snapshot, n_valid


def _get_v_range(data: np.ndarray) -> Tuple[float, float]:
    ma = np.percentile(np.abs(data), CLIM_PERCENTILE)
    return -ma, ma


def _get_ytick_labels(
    surface_channel: int, channels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, str]:
    n = len(channels)
    step = max(1, n // 5)
    yticks = list(np.arange(n)[::step])
    depths = ((surface_channel - channels) * 10).astype(int)
    yticklabels = list(depths[::step])
    zero_idx = int(np.argmin(np.abs(depths)))
    if zero_idx not in yticks:
        yticks.append(zero_idx)
        yticklabels.append(depths[zero_idx])
        order = np.argsort(yticks)
        yticks = [yticks[i] for i in order]
        yticklabels = [yticklabels[i] for i in order]
    return yticks, yticklabels, "Depth [µm]"


def _get_xtick_labels(
    window_ms: Tuple[float, float], srate: float
) -> Tuple[list, list]:
    n_samp = int((window_ms[1] - window_ms[0]) * 1e-3 * srate)
    pos_zero = int(-window_ms[0] * 1e-3 * srate)
    return [0, pos_zero, n_samp - 1], [int(window_ms[0]), 0, int(window_ms[1])]


def _plot_heatmap_panel(
    ax,
    snapshot,
    channels,
    surface_channel,
    window_ms,
    srate,
    vmin,
    vmax,
    title,
    cmap="RdBu_r",
):
    im = ax.imshow(
        snapshot,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    yticks, yticklabels, ylabel = _get_ytick_labels(surface_channel, channels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel(ylabel, fontsize=12)
    xticks, xticklabels = _get_xtick_labels(window_ms, srate)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Time [ms]", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    return im


def _plot_traces_panel(ax, snapshot, window_ms, attenuation: float = 80.0):
    n_ch = snapshot.shape[0]
    step = max(1, n_ch // 50)  # show ~50 traces
    time_axis = np.linspace(window_ms[0], window_ms[1], snapshot.shape[1])
    for i in range(0, n_ch, step):
        ax.plot(time_axis, snapshot[i] / attenuation + i, "k-", linewidth=0.5)
    ax.set_xlim(window_ms)
    ax.set_ylim([-1, n_ch])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_sta(
    awake_snap: np.ndarray,
    keta_snap: np.ndarray,
    channels: np.ndarray,
    surface_channel: int,
    window_ms: Tuple[float, float],
    srate: float,
    session_name: str,
    save_path: Path,
    trace_attenuation: float = 80.0,
):
    """four-panel STA figure: awake heatmap | traces | keta heatmap | traces."""
    combined = np.concatenate([awake_snap.ravel(), keta_snap.ravel()])
    vmin, vmax = _get_v_range(combined)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 8),
        gridspec_kw={"width_ratios": [3, 1, 3, 1], "wspace": 0.05},
    )

    im = _plot_heatmap_panel(
        axes[0],
        awake_snap,
        channels,
        surface_channel,
        window_ms,
        srate,
        vmin,
        vmax,
        "Awake",
    )
    _plot_traces_panel(axes[1], awake_snap, window_ms, trace_attenuation)

    _plot_heatmap_panel(
        axes[2],
        keta_snap,
        channels,
        surface_channel,
        window_ms,
        srate,
        vmin,
        vmax,
        "Ketamine",
    )
    _plot_traces_panel(axes[3], keta_snap, window_ms, trace_attenuation)

    fig.colorbar(im, ax=axes[3], label="Voltage [µV]", fraction=0.3, pad=0.05)
    axes[2].set_ylabel("")
    axes[2].set_yticks([])

    fig.suptitle(session_name, fontsize=14)
    plt.savefig(
        save_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", transparent=True
    )
    plt.savefig(
        save_path.with_suffix(".png"), dpi=300, bbox_inches="tight", transparent=True
    )
    plt.close()


def _get_surface_channel(meta_txt_path: Path) -> int:
    """return surface channel from 'sur <channel>' line in meta.txt."""
    for line in meta_txt_path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "sur":
            return int(parts[1])
    raise ValueError(f"'sur' field not found in {meta_txt_path}")


def process_session(session_dir: Path):
    rec_dirs = list(session_dir.glob("*_imec0"))
    if not rec_dirs:
        print(f"  [SKIP] No *_imec0 dir in {session_dir}")
        return
    rec_dir = rec_dirs[0]

    bin_files = [f for f in rec_dir.iterdir() if f.name.endswith(".ap.bin")]
    if not bin_files:
        print(f"  [SKIP] No .ap.bin in {rec_dir}")
        return
    bin_path = bin_files[0]

    output_dir = rec_dir / OUTPUT_DIR_NAME
    stim_csv = output_dir / "stim_amplitudes.csv"

    meta = _read_meta(bin_path)
    srate = _samp_rate(meta)
    fI2V = _int2volts(meta)
    gains = _ap_gains(meta)
    n_ap, _, _ = _chan_counts_im(meta)

    surface_channel = _get_surface_channel(rec_dir / "meta.txt")

    channels = np.arange(n_ap)
    if DEPTH_RANGE_UM is not None:
        depths = (surface_channel - channels) * 10
        lo_d, hi_d = DEPTH_RANGE_UM
        mask = np.ones(len(channels), dtype=bool)
        if lo_d is not None:
            mask &= depths >= lo_d
        if hi_d is not None:
            mask &= depths <= hi_d
        channels = channels[mask]

    stim_df = pd.read_csv(stim_csv)
    lo, hi = STIM_AMPLITUDE_RANGE
    if lo is not None:
        stim_df = stim_df[stim_df["amplitude_v"] >= lo]
    if hi is not None:
        stim_df = stim_df[stim_df["amplitude_v"] <= hi]

    awake_onsets = stim_df[stim_df["brain_state"] == "awake"]["onset_time_s"].values
    keta_onsets = stim_df[stim_df["brain_state"] == "ketamine"]["onset_time_s"].values

    if len(awake_onsets) == 0 or len(keta_onsets) == 0:
        print(
            f"  [SKIP] Insufficient trials after amplitude filter in {session_dir.name}"
        )
        return

    raw = _make_memmap(bin_path, meta)

    awake_samples = (awake_onsets * srate).astype(int)
    keta_samples = (keta_onsets * srate).astype(int)

    print(f"  Awake: {len(awake_samples)} trials | Keta: {len(keta_samples)} trials")

    awake_snap, n_awake = extract_voltage_snapshot(
        raw,
        awake_samples,
        channels,
        WINDOW_MS,
        srate,
        fI2V,
        gains,
        smooth_sigma=SMOOTH_SIGMA,
    )
    keta_snap, n_keta = extract_voltage_snapshot(
        raw,
        keta_samples,
        channels,
        WINDOW_MS,
        srate,
        fI2V,
        gains,
        smooth_sigma=SMOOTH_SIGMA,
    )
    print(f"  Averaged {n_awake} awake / {n_keta} keta trials")

    save_path = output_dir / "raw_sta"
    plot_sta(
        awake_snap,
        keta_snap,
        channels,
        surface_channel,
        WINDOW_MS,
        srate,
        session_dir.name,
        save_path,
    )
    print(f"  Saved → {save_path}.pdf/.png")


def main():
    with open(SESSIONS_TOML, "rb") as f:
        cfg = tomllib.load(f)
    dirs = cfg["sessions"]["dirs"]
    print(f"Processing {len(dirs)} sessions")
    for d in dirs:
        print(f"\n{'=' * 60}\n{d}")
        try:
            process_session(Path(d))
        except Exception as e:
            print(f"  [ERROR] {e}")


if __name__ == "__main__":
    main()
