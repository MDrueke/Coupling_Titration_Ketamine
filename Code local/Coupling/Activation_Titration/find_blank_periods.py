"""detect blank/data-loss periods in raw Neuropixels AP data.

Criterion: mean RMS across CHANNELS drops below RMS_THRESHOLD for at least
MIN_BLANK_DURATION_S seconds. Saves onset/offset times to blank_periods.csv
in the session's output directory.

Efficiency:
- Only CHANNELS are read (4 channels << 385).
- Non-overlapping WIN_S windows — no overlap needed for on/offset detection.
- Sessions processed in parallel (one worker per session).
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tomllib

# ── parameters ────────────────────────────────────────────────────────────────
CHANNELS = list(range(0, 40, 2))  # 20 channels: every second channel from 0 to 38
ABS_THRESHOLD = 1000.0  # µV — mean |signal| must exceed this during blank
STD_THRESHOLD = 100.0  # µV — signal std must be below this during blank
MIN_BLANK_DURATION_S = 0.5  # minimum contiguous blank duration to report
WIN_S = 0.05  # window length in s (50 ms, non-overlapping)
N_WORKERS = 5  # parallel session workers

# set to True to run on all sessions in sessions.toml; False = test recording only
USE_SESSIONS_TOML = False
TEST_DIR = Path(
    "/mnt/DATA/Coupling/2026_04_06_CouplingUre_3_g0/2026_04_06_CouplingUre_3_g0_imec0"
)


# ── raw data helpers (shared with plot_raw_average.py) ────────────────────────
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
    scale: np.ndarray,  # per-channel µV scale factor (fI2V / gain * 1e6)
    win_samp: int,
) -> tuple[np.ndarray, np.ndarray]:
    """return (mean_abs, std) averaged across channels in non-overlapping windows (µV)."""
    n_samp = raw.shape[1]
    n_win = n_samp // win_samp
    data = raw[np.ix_(channels, np.arange(n_win * win_samp))].astype(np.float32)
    data *= scale[:, np.newaxis]
    # (n_chan, n_win, win_samp)
    data = data.reshape(len(channels), n_win, win_samp)
    mean_abs = np.abs(data).mean(axis=2).mean(axis=0)  # (n_win,)
    std = data.std(axis=2).mean(axis=0)  # (n_win,)
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


# ── per-recording entry point ─────────────────────────────────────────────────
def process_recording(rec_dir: Path):
    bin_files = [f for f in rec_dir.iterdir() if f.name.endswith(".ap.bin")]
    if not bin_files:
        return
    bin_path = bin_files[0]

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

    out_dir = rec_dir / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "blank_periods.csv"

    df = (
        pd.DataFrame(periods, columns=["onset_s", "offset_s"])
        if periods
        else pd.DataFrame(columns=["onset_s", "offset_s"])
    )
    df["duration_s"] = (
        df["offset_s"] - df["onset_s"] if periods else pd.Series(dtype=float)
    )
    df.to_csv(out_path, index=False)


# ── session discovery ─────────────────────────────────────────────────────────
def find_rec_dir(session_dir: Path) -> Path | None:
    hits = list(session_dir.glob("*_imec0"))
    return hits[0] if hits else None


def _worker(rec_dir: Path):
    try:
        process_recording(rec_dir)
    except Exception:
        pass


def main():
    if USE_SESSIONS_TOML:
        with open("sessions.toml", "rb") as f:
            cfg = tomllib.load(f)
        rec_dirs = []
        for d in cfg["sessions"]["dirs"]:
            rec_dir = find_rec_dir(Path(d))
            if rec_dir:
                rec_dirs.append(rec_dir)
    else:
        rec_dirs = [TEST_DIR]

    if len(rec_dirs) == 1:
        _worker(rec_dirs[0])
    else:
        with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(rec_dirs))) as ex:
            list(ex.map(_worker, rec_dirs))


if __name__ == "__main__":
    main()
