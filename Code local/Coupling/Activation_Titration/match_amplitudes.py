"""match NIDQ pulse onset times to amplitudes from WaveformSequence.csv."""

from pathlib import Path

import numpy as np
import pandas as pd

_TEAL   = "\033[38;2;187;230;228m"
_ORANGE = "\033[38;2;255;149;5m"
_RESET  = "\033[0m"

MAX_TOL_S         = 1.0    # max |AP − WF| residual to accept a match (s)
HIST_BIN_S        = 0.05   # histogram bin width for offset estimation (s)
QC_MAX_RESIDUAL_S = 0.150  # error if any matched residual exceeds this (s)
QC_MAX_MEAN_ABS_S = 0.030  # error if |mean residual| exceeds this — catches bad offset bin
QC_MAX_MISS_FRAC  = 0.15   # warning if fraction of unmatched WF entries exceeds this


def match_amplitudes(ap_onsets: np.ndarray,
                     waveform_csv: str,
                     block_label: str,
                     min_amplitude_v: float = None,
                     diag_dir: Path = None) -> pd.DataFrame:
    """
    match aligned AP onsets to WaveformSequence amplitudes.

    amp[j] in WaveformSequence is queued at time[j] for the pulse at time[j+1].
    Returns one row per matched pulse with columns: onset_time_s, amplitude_v.
    """
    df_wf    = pd.read_csv(waveform_csv)
    times_all = df_wf["Time(s)"].values
    amps_all  = df_wf["Amplitude(V)"].values

    # amp[j] is queued at time[j] for the pulse that fires at time[j+1].
    # Filter on original indices so amp[j] always pairs with time[j+1] from the
    # original CSV, not with the next filtered time (which would skip gaps).
    keep = amps_all[:-1] >= min_amplitude_v if min_amplitude_v is not None else slice(None)
    n_dropped = int((amps_all < min_amplitude_v).sum()) if min_amplitude_v is not None else 0
    if n_dropped:
        print(f"{_TEAL}  Dropped {n_dropped} sub-threshold WF entries (amp < {min_amplitude_v} V){_RESET}")
    wf_times = times_all[1:][keep]
    wf_amps  = amps_all[:-1][keep]

    # last row: amp[-1] queued for a pulse beyond time[-1]; estimate onset via median ITI.
    # its residual reflects ITI variance (not matching error) so it is excluded from QC.
    has_extrapolated = min_amplitude_v is None or amps_all[-1] >= min_amplitude_v
    if has_extrapolated:
        iti = float(np.median(np.diff(times_all)))
        wf_times = np.append(wf_times, times_all[-1] + iti)
        wf_amps  = np.append(wf_amps,  amps_all[-1])

    offset = _estimate_offset(wf_times, ap_onsets)
    wf_ap  = wf_times + offset
    print(f"{_TEAL}  WF→AP offset: {offset:.4f} s{_RESET}")

    matched_ap_idx, residuals = _nn_match(wf_ap, ap_onsets, MAX_TOL_S)

    matched_mask  = matched_ap_idx >= 0
    n_matched     = int(matched_mask.sum())
    n_missed_wf   = int((~matched_mask).sum())
    ap_used       = set(matched_ap_idx[matched_mask].tolist())
    n_spurious_ap = len(ap_onsets) - len(ap_used)

    print(f"{_TEAL}  Matched {n_matched} / {len(wf_times)}  "
          f"({n_missed_wf} missed WF, {n_spurious_ap} spurious det){_RESET}")

    # write diagnostics before QC so files exist even if QC fails
    if diag_dir is not None:
        spurious_ap_times = ap_onsets[
            np.array([i for i in range(len(ap_onsets)) if i not in ap_used])
        ] if n_spurious_ap > 0 else np.array([])
        _write_stim_times(
            Path(diag_dir), block_label,
            wf_times, wf_amps, offset, matched_mask,
            ap_onsets, matched_ap_idx, spurious_ap_times, has_extrapolated,
        )

    # quality checks (exclude extrapolated last entry — its residual = ITI variance)
    qc_mask = matched_mask.copy()
    if has_extrapolated and matched_mask[-1]:
        qc_mask[-1] = False
    r = residuals[qc_mask]
    if len(r):
        max_r     = float(np.abs(r).max())
        mean_r    = float(np.abs(r.mean()))
        miss_frac = n_missed_wf / len(wf_times)
        if max_r > QC_MAX_RESIDUAL_S:
            outlier_idx = np.where(np.abs(r) > QC_MAX_RESIDUAL_S)[0]
            print(f"{_ORANGE}  WARNING: max |residual| = {max_r*1e3:.1f} ms "
                  f"> {QC_MAX_RESIDUAL_S*1e3:.0f} ms — check outlier(s) below:{_RESET}")
            for i in outlier_idx:
                t = float(ap_onsets[matched_ap_idx[matched_mask][i]])
                print(f"{_ORANGE}    t={t:.3f} s  residual={r[i]*1e3:.1f} ms  "
                      f"amp={wf_amps[matched_mask][i]:.4f} V{_RESET}")
        if mean_r > QC_MAX_MEAN_ABS_S:
            print(f"{_ORANGE}  WARNING: |mean residual| = {mean_r*1e3:.1f} ms "
                  f"> {QC_MAX_MEAN_ABS_S*1e3:.0f} ms — histogram offset may be in wrong bin.{_RESET}")
        if miss_frac > QC_MAX_MISS_FRAC:
            print(f"{_ORANGE}  WARNING: {miss_frac*100:.1f}% WF entries unmatched "
                  f"(threshold {QC_MAX_MISS_FRAC*100:.0f}%) — check detection or block split.{_RESET}")
        if n_spurious_ap > 0:
            print(f"{_ORANGE}  WARNING: {n_spurious_ap} spurious AP detections — "
                  f"unexpected signal in stim channel or wrong block boundaries.{_RESET}")
        print(f"{_TEAL}  Residual: mean={r.mean()*1e3:.2f} ms  "
              f"std={r.std()*1e3:.2f} ms  max|r|={max_r*1e3:.2f} ms{_RESET}")

    result = pd.DataFrame({
        "onset_time_s": ap_onsets[matched_ap_idx[matched_mask]],
        "amplitude_v":  wf_amps[matched_mask],
    })

    return result


def _estimate_offset(wf_times: np.ndarray, ap_onsets: np.ndarray) -> float:
    """all-pairs histogram estimator of WF→AP time offset.

    With N≈600 pulses the peak bin accumulates one vote per correct pair and is
    separated from any wrong-pairing clusters by at least one ITI (~2–4 s).
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
    """nearest-neighbour match of WF entries (shifted to AP clock) to AP onsets.

    Returns matched_ap_idx (index into ap_onsets, -1 if unmatched) and
    residuals (ap_onsets[match] − wf_ap, NaN where unmatched).
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
                      ap_onsets, matched_ap_idx, spurious_ap_times, has_extrapolated):
    """write stim_times_{block}.csv for manual NIDQ inspection."""
    ap_times_out = np.full(len(wf_times), np.nan)
    ap_times_out[matched_mask] = ap_onsets[matched_ap_idx[matched_mask]]

    statuses = ["matched" if m else "missed_wf" for m in matched_mask]
    if has_extrapolated and matched_mask[-1]:
        statuses[-1] = "matched_extrap"

    rows = pd.DataFrame({
        "expected_ap_s":  wf_times + offset,
        "wf_amplitude_v": wf_amps,
        "actual_ap_s":    ap_times_out,
        "status":         statuses,
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
    print(f"{_TEAL}  Stim times → {out_path.name}{_RESET}")
