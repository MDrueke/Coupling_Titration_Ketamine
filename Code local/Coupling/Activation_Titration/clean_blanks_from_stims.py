"""detect blank/data-loss periods in raw Neuropixels AP data and remove
the corresponding stimulus trials from stim_amplitudes.csv.

Detection criterion: mean |signal| > ABS_THRESHOLD µV AND std < STD_THRESHOLD µV
for at least MIN_BLANK_DURATION_S seconds (signal stuck at high DC offset).
"""

from pathlib import Path
import tomllib

import numpy as np
import pandas as pd

# ── parameters ────────────────────────────────────────────────────────────────
CHANNELS = list(range(0, 40, 2))   # 20 channels: every second channel from 0 to 38
ABS_THRESHOLD = 1000.0             # µV — mean |signal| must exceed this during blank
STD_THRESHOLD = 100.0              # µV — signal std must be below this during blank
MIN_BLANK_DURATION_S = 0.5         # minimum contiguous blank duration to report
WIN_S = 0.05                       # window length in s (50 ms, non-overlapping)


# ── raw data helpers ──────────────────────────────────────────────────────────
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


def _make_memmap(bin_path: Path, meta: dict) -> np.ndarray:
    n_chan = int(meta["nSavedChans"])
    n_samp = int(meta["fileSizeBytes"]) // (2 * n_chan)
    return np.memmap(
        bin_path, dtype="int16", mode="r", shape=(n_chan, n_samp), offset=0, order="F"
    )


# ── detection ─────────────────────────────────────────────────────────────────
def compute_window_metrics(
    raw: np.ndarray,
    channels: np.ndarray,
    scale: np.ndarray,
    win_samp: int,
) -> tuple[np.ndarray, np.ndarray]:
    """return (mean_abs, std) averaged across channels in non-overlapping windows (µV)."""
    n_samp = raw.shape[1]
    n_win = n_samp // win_samp
    data = raw[np.ix_(channels, np.arange(n_win * win_samp))].astype(np.float32)
    data *= scale[:, np.newaxis]
    data = data.reshape(len(channels), n_win, win_samp)
    mean_abs = np.abs(data).mean(axis=2).mean(axis=0)
    std = data.std(axis=2).mean(axis=0)
    return mean_abs, std


def find_blank_periods(
    mean_abs: np.ndarray,
    std: np.ndarray,
    win_s: float,
    abs_threshold: float,
    std_threshold: float,
    min_duration_s: float,
) -> list[tuple[float, float]]:
    """return (onset_s, offset_s) where signal is high and flat for >= min_duration_s."""
    blank = (mean_abs > abs_threshold) & (std < std_threshold)
    min_wins = int(np.ceil(min_duration_s / win_s))
    periods = []
    i = 0
    n = len(blank)
    while i < n:
        if blank[i]:
            j = i
            while j < n and blank[j]:
                j += 1
            if (j - i) >= min_wins:
                periods.append((i * win_s, j * win_s))
            i = j
        else:
            i += 1
    return periods


# ── per-session processing ────────────────────────────────────────────────────
def process_session(session_dir: Path) -> str:
    """detect blank periods, remove affected stims, return status string."""
    rec_dirs = list(session_dir.glob("*_imec0"))
    if not rec_dirs:
        return f"{session_dir.name}: no *_imec0 dir"
    rec_dir = rec_dirs[0]

    bin_files = [f for f in rec_dir.iterdir() if f.name.endswith(".ap.bin")]
    if not bin_files:
        return f"{session_dir.name}: no .ap.bin"
    bin_path = bin_files[0]

    stim_path = rec_dir / "output" / "stim_amplitudes.csv"
    if not stim_path.exists():
        return f"{session_dir.name}: no stim_amplitudes.csv"

    meta = _read_meta(bin_path)
    sr = _samp_rate(meta)
    fI2V = _int2volts(meta)
    gains = _ap_gains(meta)
    raw = _make_memmap(bin_path, meta)

    chans = np.array(CHANNELS)
    scale = (fI2V / gains[chans] * 1e6).astype(np.float32)
    win_samp = int(WIN_S * sr)

    mean_abs, std = compute_window_metrics(raw, chans, scale, win_samp)
    periods = find_blank_periods(
        mean_abs, std, WIN_S, ABS_THRESHOLD, STD_THRESHOLD, MIN_BLANK_DURATION_S
    )

    # save blank periods
    blank_path = rec_dir / "output" / "blank_periods.csv"
    df_blank = pd.DataFrame(periods, columns=["onset_s", "offset_s"]) if periods else pd.DataFrame(columns=["onset_s", "offset_s"])
    df_blank["duration_s"] = df_blank["offset_s"] - df_blank["onset_s"] if periods else pd.Series(dtype=float)
    df_blank.to_csv(blank_path, index=False)

    # remove stims that fall within any blank period
    stim_df = pd.read_csv(stim_path)
    n_before = len(stim_df)

    if periods:
        onsets = stim_df["onset_time_s"].values
        in_blank = np.zeros(len(onsets), dtype=bool)
        for on, off in periods:
            in_blank |= (onsets >= on) & (onsets <= off)
        stim_df = stim_df[~in_blank]

    n_removed = n_before - len(stim_df)
    stim_df.to_csv(stim_path, index=False)

    return f"{session_dir.name}: {len(periods)} blank period(s), {n_removed}/{n_before} stims removed"


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    with open("sessions.toml", "rb") as f:
        cfg = tomllib.load(f)
    dirs = [Path(d) for d in cfg["sessions"]["dirs"]]

    for d in dirs:
        print(f"{d.name} ...", flush=True)
        result = process_session(d)
        print(f"  {result}")


if __name__ == "__main__":
    main()
