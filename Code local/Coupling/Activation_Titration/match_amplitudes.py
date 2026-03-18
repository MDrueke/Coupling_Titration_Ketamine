"""
Match NIDQ pulse onset times to amplitudes from WaveformSequence.csv.

Step 1: estimate WF→AP time offset via all-pairs histogram.
        Every correct WF↔AP pair contributes one count to the peak bin at the
        true offset; wrong pairings scatter across bins at ±k·ITI. With ~600
        correct pairs the peak is unambiguous regardless of missing pulses.
Step 2: nearest-neighbour match each WF entry to its closest AP onset within
        MAX_TOL_S. Sub-threshold entries (filtered by min_amplitude_v) are dropped
        before matching; remaining unmatched entries are classified as missed_wf.
"""

from pathlib import Path

import numpy as np
import pandas as pd

MAX_TOL_S         = 1.0   # max |AP − WF| residual to accept a match (s)
HIST_BIN_S        = 0.05  # histogram bin width for offset estimation (s)
QC_MAX_RESIDUAL_S = 0.150 # error if any matched residual exceeds this (s)
QC_MAX_MEAN_ABS_S = 0.030 # error if |mean residual| exceeds this — catches bad offset bin
QC_MAX_MISS_FRAC  = 0.15  # warning if fraction of unmatched WF entries exceeds this


def match_amplitudes(ap_onsets: np.ndarray,
                     waveform_csv: str,
                     block_label: str,
                     min_amplitude_v: float = None,
                     diag_dir: Path = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    ap_onsets : np.ndarray
        Aligned pulse onset times in AP-clock seconds.
    waveform_csv : str
        Path to WaveformSequence CSV with Time(s) and Amplitude(V) columns.
    block_label : str
        Short label used for output filenames (e.g. "awake", "keta").
    min_amplitude_v : float, optional
        Drop WF rows below this amplitude before matching.
    diag_dir : Path, optional
        If given, write stim_times_{block_label}.csv here.

    Returns
    -------
    pd.DataFrame  columns: onset_time_s, amplitude_v
        One row per matched pulse; missed WF entries are excluded.
    """
    df_wf = pd.read_csv(waveform_csv)
    if min_amplitude_v is not None:
        n_dropped = int((df_wf["Amplitude(V)"] < min_amplitude_v).sum())
        if n_dropped:
            print(f"  Dropped {n_dropped} sub-threshold WF entries (amp < {min_amplitude_v} V)")
        df_wf = df_wf[df_wf["Amplitude(V)"] >= min_amplitude_v].reset_index(drop=True)

    wf_times = df_wf["Time(s)"].values
    wf_amps  = df_wf["Amplitude(V)"].values

    offset = _estimate_offset(wf_times, ap_onsets)
    wf_ap  = wf_times + offset
    print(f"  WF→AP offset: {offset:.4f} s")

    matched_ap_idx, residuals = _nn_match(wf_ap, ap_onsets, MAX_TOL_S)

    matched_mask  = matched_ap_idx >= 0
    n_matched     = int(matched_mask.sum())
    n_missed_wf   = int((~matched_mask).sum())
    ap_used       = set(matched_ap_idx[matched_mask].tolist())
    n_spurious_ap = len(ap_onsets) - len(ap_used)

    print(f"  Matched {n_matched} / {len(wf_times)}  "
          f"({n_missed_wf} missed WF, {n_spurious_ap} spurious det)")

    # --- quality checks ---
    r = residuals[matched_mask]
    if len(r):
        max_r    = float(np.abs(r).max())
        mean_r   = float(np.abs(r.mean()))
        miss_frac = n_missed_wf / len(wf_times)
        if max_r > QC_MAX_RESIDUAL_S:
            raise ValueError(
                f"Alignment QC failed [{block_label}]: max |residual| = {max_r*1e3:.1f} ms "
                f"> {QC_MAX_RESIDUAL_S*1e3:.0f} ms — likely a bad match."
            )
        if mean_r > QC_MAX_MEAN_ABS_S:
            raise ValueError(
                f"Alignment QC failed [{block_label}]: |mean residual| = {mean_r*1e3:.1f} ms "
                f"> {QC_MAX_MEAN_ABS_S*1e3:.0f} ms — histogram offset may be in wrong bin."
            )
        if miss_frac > QC_MAX_MISS_FRAC:
            print(f"  WARNING: {miss_frac*100:.1f}% WF entries unmatched "
                  f"(threshold {QC_MAX_MISS_FRAC*100:.0f}%) — check detection or block split.")
        if n_spurious_ap > 0:
            print(f"\033[91m  WARNING: {n_spurious_ap} spurious AP detections — "
                  "unexpected signal in stim channel or wrong block boundaries.\033[0m")
        print(f"  Residual: mean={r.mean()*1e3:.2f} ms  "
              f"std={r.std()*1e3:.2f} ms  max|r|={max_r*1e3:.2f} ms")

    # --- build result (matched pulses only) ---
    result = pd.DataFrame({
        "onset_time_s": ap_onsets[matched_ap_idx[matched_mask]],
        "amplitude_v":  wf_amps[matched_mask],
    })

    if diag_dir is not None:
        spurious_ap_times = ap_onsets[
            np.array([i for i in range(len(ap_onsets)) if i not in ap_used])
        ] if n_spurious_ap > 0 else np.array([])
        _write_stim_times(
            Path(diag_dir), block_label,
            wf_times, wf_amps, offset, matched_mask,
            ap_onsets, matched_ap_idx, spurious_ap_times,
        )

    return result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _estimate_offset(wf_times: np.ndarray, ap_onsets: np.ndarray) -> float:
    """
    All-pairs histogram offset estimator.

    Computes every pairwise difference ap[i] − wf[j] and finds the peak bin.
    With N≈600 pulses the matrix is 600×600 ≈ 360 k values — trivial cost.
    The peak bin accumulates one vote per correct pair and is separated from
    any wrong-pairing clusters by at least one ITI (~2–4 s).
    """
    diffs  = (ap_onsets[:, None] - wf_times[None, :]).ravel()
    lo, hi = diffs.min() - 1.0, diffs.max() + 1.0 + HIST_BIN_S
    bins   = np.arange(lo, hi, HIST_BIN_S)
    counts, edges = np.histogram(diffs, bins=bins)
    peak   = int(np.argmax(counts))
    return 0.5 * (edges[peak] + edges[peak + 1])


def _nn_match(wf_ap: np.ndarray,
              ap_onsets: np.ndarray,
              max_tol: float) -> tuple[np.ndarray, np.ndarray]:
    """
    For each WF entry (already shifted to AP clock), find the nearest
    unmatched AP onset within max_tol.

    Returns
    -------
    matched_ap_idx : int array, shape (n_wf,)
        Index into ap_onsets for each WF entry; −1 if no match within tolerance.
    residuals : float array, shape (n_wf,)
        ap_onsets[match] − wf_ap  (positive = AP late); NaN where unmatched.
    """
    order = np.argsort(ap_onsets)
    ap_s  = ap_onsets[order]

    matched = np.full(len(wf_ap), -1, dtype=int)
    resid   = np.full(len(wf_ap), np.nan)
    used    = set()

    for j, t in enumerate(wf_ap):
        pos = int(np.searchsorted(ap_s, t))
        best_d, best_orig = max_tol, -1
        for k in (pos - 1, pos):
            if 0 <= k < len(ap_s):
                orig = int(order[k])
                if orig not in used:
                    d = abs(ap_s[k] - t)
                    if d < best_d:
                        best_d, best_orig = d, orig
        if best_orig >= 0:
            matched[j] = best_orig
            resid[j]   = ap_onsets[best_orig] - t
            used.add(best_orig)

    return matched, resid


def _write_stim_times(diag_dir: Path, block_label: str,
                      wf_times, wf_amps, offset, matched_mask,
                      ap_onsets, matched_ap_idx, spurious_ap_times):
    """Write stim_times_{block}.csv for manual NIDQ inspection."""
    ap_times_out = np.full(len(wf_times), np.nan)
    ap_times_out[matched_mask] = ap_onsets[matched_ap_idx[matched_mask]]

    rows = pd.DataFrame({
        "expected_ap_s":  wf_times + offset,
        "wf_amplitude_v": wf_amps,
        "actual_ap_s":    ap_times_out,
        "status":         ["matched" if m else "missed_wf" for m in matched_mask],
    })

    if len(spurious_ap_times) > 0:
        spur_rows = pd.DataFrame({
            "expected_ap_s":  np.nan,
            "wf_amplitude_v": np.nan,
            "actual_ap_s":    spurious_ap_times,
            "status":         "spurious_ap",
        })
        rows = pd.concat([rows, spur_rows], ignore_index=True)

    df = rows.sort_values("actual_ap_s", na_position="last").reset_index(drop=True)

    out_path = diag_dir / f"stim_times_{block_label}.csv"
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  Stim times → {out_path.name}")
