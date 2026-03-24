"""data stream alignment using sync pulses.

aligns multi-device recordings (e.g., NIDQ + neuropixels) by detecting shared
sync pulses and correcting for temporal drift between devices.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import re

try:
    import mtscomp
    HAS_MTSCOMP = True
except ImportError:
    HAS_MTSCOMP = False
    print("Warning: mtscomp not available. Compressed .cbin files cannot be read.")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


_TEAL   = "\033[38;2;187;230;228m"
_ORANGE = "\033[38;2;255;149;5m"
_RESET  = "\033[0m"


def read_meta(bin_path: Path) -> dict:
    """parse SGLX meta file into dictionary. works for .bin and .cbin."""
    meta_path = Path(bin_path).with_suffix(".meta")
    meta_dict = {}
    with open(meta_path) as meta_file:
        for line in meta_file:
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if val and re.fullmatch(r"[0-9,.]*", val) and val.count(".") < 2:
                    parsed = [float(v) for v in val.split(",")]
                    val = parsed[0] if len(parsed) == 1 else parsed
                meta_dict[key] = val
    return meta_dict


def read_channel_from_cbin(cbin_path: Path, channel_idx: int) -> np.ndarray:
    """read a single channel from a .cbin file without writing to disk."""
    if not HAS_MTSCOMP:
        raise ImportError("mtscomp required to read .cbin files.")
    r = mtscomp.Reader()
    r.open(cbin_path)
    n_channels = r.n_channels
    if channel_idx < 0:
        channel_idx = n_channels + channel_idx
    chunks = []
    chunk_iter = list(r.iter_chunks())
    it = tqdm(chunk_iter, desc=f"  Reading {Path(cbin_path).name}", unit="chunk") if HAS_TQDM else chunk_iter
    for chunk_idx, chunk_start, chunk_length in it:
        chunk = r.read_chunk(chunk_idx, chunk_start, chunk_length)
        chunks.append(chunk[:, channel_idx].copy())
    r.close()
    return np.concatenate(chunks)


def make_memmap(bin_path: Path, meta: dict) -> np.memmap:
    """memory-map binary data file."""
    n_chan = int(meta["nSavedChans"])
    n_samp = int(int(meta["fileSizeBytes"]) / (2 * n_chan))
    return np.memmap(bin_path, dtype=np.int16, mode="r", shape=(n_chan, n_samp), order="F")


def unpack_bits(channel_data: np.ndarray) -> np.ndarray:
    """unpack all 16 bits from a 1D int16 channel array. returns (16, n_samples) uint8."""
    dig = np.zeros((16, len(channel_data)), dtype=np.uint8)
    for bit in range(16):
        dig[bit, :] = (channel_data >> bit) & 0x01
    return dig


def extract_digital_channel(raw_data: np.memmap, channel_idx: int) -> np.ndarray:
    """extract and unpack a digital channel from a 2D memmap. returns (16, n_samples) uint8."""
    if channel_idx < 0:
        channel_idx = raw_data.shape[0] + channel_idx
    return unpack_bits(raw_data[channel_idx, :])


def merge_nearby_pulses(onset_samples: np.ndarray, offset_samples: np.ndarray,
                       sample_rate: float, max_gap_ms: float = 1.0,
                       target_duration_ms: float = 28.0,
                       tolerance_ms: float = 5.0) -> tuple:
    """merge clusters of pulses separated by short gaps if the cluster span matches target duration."""
    if len(onset_samples) == 0:
        return onset_samples, offset_samples

    max_gap_samples = int((max_gap_ms / 1000) * sample_rate)
    min_span = (target_duration_ms - tolerance_ms) / 1000 * sample_rate
    max_span = (target_duration_ms + tolerance_ms) / 1000 * sample_rate

    merged_onsets = []
    merged_offsets = []

    i = 0
    while i < len(onset_samples):
        j = i
        cluster_end = offset_samples[i]
        while j + 1 < len(onset_samples):
            gap = onset_samples[j + 1] - offset_samples[j]
            if gap <= max_gap_samples:
                j += 1
                cluster_end = offset_samples[j]
            else:
                break

        if j > i:
            span = cluster_end - onset_samples[i]
            if min_span <= span <= max_span:
                merged_onsets.append(onset_samples[i])
                merged_offsets.append(cluster_end)
                i = j + 1
                continue
            for k in range(i, j + 1):
                merged_onsets.append(onset_samples[k])
                merged_offsets.append(offset_samples[k])
            i = j + 1
        else:
            merged_onsets.append(onset_samples[i])
            merged_offsets.append(offset_samples[i])
            i += 1

    return np.array(merged_onsets), np.array(merged_offsets)


def extract_pulses_with_duration(pulse_data: np.ndarray, sample_rate: float,
                                 target_duration_ms: float = None,
                                 tolerance_ms: float = 2.0,
                                 merge_gap_ms: float = 0.0) -> np.ndarray:
    """extract pulse onset times, optionally filtering by duration.

    if target_duration_ms is None, every rising edge is returned.
    """
    data = pulse_data.astype(np.int8)
    diffs = np.diff(data, prepend=0, append=0)

    onset_samples = np.where(diffs == 1)[0]
    offset_samples = np.where(diffs == -1)[0]

    assert len(onset_samples) == len(offset_samples), "Onset/offset mismatch"

    if target_duration_ms is None:
        return onset_samples / sample_rate

    if merge_gap_ms > 0:
        onset_samples, offset_samples = merge_nearby_pulses(
            onset_samples, offset_samples, sample_rate,
            max_gap_ms=merge_gap_ms,
            target_duration_ms=target_duration_ms,
            tolerance_ms=tolerance_ms
        )

    durations_ms = (offset_samples - onset_samples) / sample_rate * 1000
    min_dur = target_duration_ms - tolerance_ms
    max_dur = target_duration_ms + tolerance_ms
    mask = (durations_ms >= min_dur) & (durations_ms <= max_dur)

    n_rejected = int((~mask).sum())
    if n_rejected > 0:
        rejected_times = onset_samples[~mask] / sample_rate
        rejected_durs = durations_ms[~mask]
        print(f"{_TEAL}  Duration filter: rejected {n_rejected} pulse(s) outside "
              f"{min_dur:.1f}–{max_dur:.1f} ms window:{_RESET}")
        for t, d in zip(rejected_times, rejected_durs):
            print(f"{_TEAL}    t={t:.3f} s  dur={d:.2f} ms{_RESET}")

    return onset_samples[mask] / sample_rate


class DataStreamAligner:
    """align multi-device recordings using sync pulses.

    corrects temporal drift between a reference stream (e.g., neuropixels AP)
    and target streams (e.g., NIDQ) using shared sync pulses.
    """

    def __init__(self, reference_file: Path, reference_sync_channel: int,
                 sync_params: dict, reference_sync_bit: int = 0,
                 cache_dir: Optional[Path] = None):
        """
        Parameters
        ----------
        reference_file : Path
            path to reference .bin or .cbin file (e.g., .ap.bin)
        reference_sync_channel : int
            channel index for sync pulses (negative indexing supported)
        sync_params : dict
            pulse extraction parameters: target_duration_ms, tolerance_ms, merge_gap_ms
        reference_sync_bit : int
            bit number within sync channel (0-15). for AP files, use 6.
        """
        self.reference_file = Path(reference_file)
        self.reference_sync_channel = reference_sync_channel
        self.reference_sync_bit = reference_sync_bit

        # pop alignment-control keys before passing sync_params to extract_pulses_with_duration
        sync_params = dict(sync_params)
        self.max_trim = sync_params.pop("max_trim", 1)
        self.check_sync_ipi = sync_params.pop("check_sync_ipi", True)
        self.sync_params = sync_params

        self.reference_meta = read_meta(self.reference_file)
        assert self.reference_meta, f"Meta file not found for {self.reference_file}"

        self.reference_sample_rate = float(self.reference_meta["imSampRate"])

        print(f"Extracting sync from reference: {self.reference_file.name}")
        cache_file = None
        if cache_dir is not None:
            cache_file = Path(cache_dir) / f"{self.reference_file.stem}_sync_ch.npy"

        if self.reference_file.suffix == ".cbin":
            if cache_file is not None and cache_file.exists():
                print(f"  Loading cached sync channel from {cache_file.name}")
                ref_channel = np.load(cache_file)
            else:
                ref_channel = read_channel_from_cbin(self.reference_file, reference_sync_channel)
                if cache_file is not None:
                    np.save(cache_file, ref_channel)
                    print(f"  Cached sync channel to {cache_file.name}")
        else:
            ref_data = make_memmap(self.reference_file, self.reference_meta)
            ch_idx = reference_sync_channel if reference_sync_channel >= 0 else ref_data.shape[0] + reference_sync_channel
            ref_channel = np.array(ref_data[ch_idx, :])
        ref_dig = unpack_bits(ref_channel)
        self._ref_sync_line = ref_dig[reference_sync_bit]

        print(f"  Using bit {reference_sync_bit} for sync")
        self.reference_sync_onsets = extract_pulses_with_duration(
            ref_dig[reference_sync_bit], self.reference_sample_rate, **self.sync_params
        )

        print(f"{_TEAL}  Found {len(self.reference_sync_onsets)} sync pulses{_RESET}")
        if self.check_sync_ipi:
            self.reference_sync_onsets = self._check_ipi(self.reference_sync_onsets, "reference")

        self.target_streams: Dict[str, dict] = {}

    def _check_ipi(self, onsets: np.ndarray, stream_name: str) -> np.ndarray:
        """check sync pulse inter-pulse intervals; warn on short IPIs, fill long gaps with synthetic pulses."""
        if len(onsets) < 2:
            return onsets

        ipis = np.diff(onsets)
        median_ipi = float(np.median(ipis))

        short_idx = np.where(ipis < 0.5 * median_ipi)[0]
        for i in short_idx:
            print(
                f"{_ORANGE}  !!! WARNING: SHORT IPI in '{stream_name}' between pulses "
                f"{i} and {i+1}: {ipis[i]*1000:.1f} ms "
                f"(expected ~{median_ipi*1000:.0f} ms) — possible spurious pulse !!!{_RESET}"
            )

        long_idx = np.where(ipis > 1.5 * median_ipi)[0]
        if len(long_idx) == 0:
            return onsets

        corrected = list(onsets)
        offset = 0
        for i in long_idx:
            adj_i = i + offset
            n_missing = round(ipis[i] / median_ipi) - 1
            for j in range(n_missing):
                corrected.insert(adj_i + j + 1, corrected[adj_i] + (j + 1) * median_ipi)
            print(
                f"{_ORANGE}  !!! WARNING: MISSING SYNC PULSE in '{stream_name}' after pulse {i} "
                f"(t={onsets[i]:.3f} s): IPI = {ipis[i]*1000:.1f} ms, "
                f"expected ~{median_ipi*1000:.0f} ms. "
                f"Inserted {n_missing} synthetic pulse(s). !!!{_RESET}"
            )
            offset += n_missing

        return np.array(corrected)

    def add_target_stream(self, target_file: Path, target_sync_channel: int,
                         stream_name: str, target_sync_bit: int = 0):
        """add a target stream to align against the reference."""
        target_file = Path(target_file)

        target_meta = read_meta(target_file)
        assert target_meta, f"Meta file not found for {target_file}"

        target_sample_rate = float(target_meta.get("niSampRate", target_meta.get("imSampRate")))

        print(f"\nExtracting sync from target '{stream_name}': {target_file.name}")
        if target_file.suffix == ".cbin":
            sync_channel_data = read_channel_from_cbin(target_file, target_sync_channel)
            target_dig = unpack_bits(sync_channel_data)
            target_data = None
        else:
            target_data = make_memmap(target_file, target_meta)
            target_dig = extract_digital_channel(target_data, target_sync_channel)

        print(f"  Using bit {target_sync_bit} for sync")
        target_sync_onsets = extract_pulses_with_duration(
            target_dig[target_sync_bit], target_sample_rate, **self.sync_params
        )

        print(f"{_TEAL}  Found {len(target_sync_onsets)} sync pulses{_RESET}")
        if self.check_sync_ipi:
            target_sync_onsets = self._check_ipi(target_sync_onsets, stream_name)

        n_ref = len(self.reference_sync_onsets)
        n_tgt = len(target_sync_onsets)
        if n_ref != n_tgt:
            diff = abs(n_ref - n_tgt)
            ref_last = self.reference_sync_onsets[-1] if n_ref > 0 else float("nan")
            tgt_last = target_sync_onsets[-1] if n_tgt > 0 else float("nan")
            ref_dur = self.reference_sync_onsets[-1] - self.reference_sync_onsets[0] if n_ref > 1 else 0
            tgt_dur = target_sync_onsets[-1] - target_sync_onsets[0] if n_tgt > 1 else 0
            if diff <= self.max_trim:
                longer = "target" if n_tgt > n_ref else "reference"
                n_keep = min(n_ref, n_tgt)

                # determine which end has the extra pulse by comparing first pulse times
                median_interval = float(np.median(np.diff(self.reference_sync_onsets)))
                first_diff = abs(target_sync_onsets[0] - self.reference_sync_onsets[0])
                trim_end = "end" if first_diff < 0.5 * median_interval else "beginning"

                if n_tgt > n_ref:
                    if trim_end == "end":
                        target_sync_onsets = target_sync_onsets[:n_keep]
                    else:
                        target_sync_onsets = target_sync_onsets[diff:]
                else:
                    if trim_end == "end":
                        self.reference_sync_onsets = self.reference_sync_onsets[:n_keep]
                    else:
                        self.reference_sync_onsets = self.reference_sync_onsets[diff:]

                if trim_end == "end":
                    if longer == "target":
                        edge_high = bool(target_dig[target_sync_bit, -1])
                    else:
                        edge_high = bool(self._ref_sync_line[-1])
                    edge_label = "last"
                else:
                    if longer == "target":
                        edge_high = bool(target_dig[target_sync_bit, 0])
                    else:
                        edge_high = bool(self._ref_sync_line[0])
                    edge_label = "first"

                truncation_msg = (
                    f"  Sync line was HIGH at {edge_label} sample of {longer} — confirms truncated/partial pulse."
                    if edge_high else
                    f"  Sync line was LOW at {edge_label} sample of {longer} — extra pulse cause unclear; check data."
                )

                print(f"{_ORANGE}  !!! WARNING: TRIMMED {diff} SYNC PULSE(S) FROM {trim_end.upper()} OF {longer.upper()} !!!")
                print(f"  Counts now match: {n_keep}{_RESET}")
                print(truncation_msg)
            else:
                raise ValueError(
                    f"Sync pulse count mismatch after IPI gap-filling!\n"
                    f"  Reference: {n_ref} pulses, last at {ref_last:.3f} s, span {ref_dur:.1f} s\n"
                    f"  Target:    {n_tgt} pulses, last at {tgt_last:.3f} s, span {tgt_dur:.1f} s\n"
                    f"  Difference: {diff} pulse(s) — too large to auto-trim (max_trim={self.max_trim})\n"
                    f"  Check for spurious pulses (short-IPI warnings above) or multiple missing pulses."
                )

        drift = self.reference_sync_onsets - target_sync_onsets
        max_drift = np.max(np.abs(drift))

        print(f"{_TEAL}  Max drift: {max_drift*1000:.2f} ms{_RESET}")
        if max_drift > 0.1:
            print(f"{_ORANGE}  WARNING: Drift exceeds 100ms threshold!{_RESET}")

        self.target_streams[stream_name] = {
            "file": target_file,
            "meta": target_meta,
            "sample_rate": target_sample_rate,
            "sync_onsets": target_sync_onsets,
            "data": target_data,
        }

    def _correct_times(self, target_times: np.ndarray, stream_name: str) -> np.ndarray:
        """apply nearest-sync correction to event times (vectorized)."""
        stream = self.target_streams[stream_name]
        target_sync = stream["sync_onsets"]

        indices = np.searchsorted(target_sync, target_times)
        indices = np.clip(indices, 0, len(target_sync) - 1)

        shifts = self.reference_sync_onsets[indices] - target_sync[indices]
        return target_times + shifts

    def align_channel(self, stream_name: str, channel_number: int,
                     pulse_params: dict, output_dir: Optional[Path] = None) -> np.ndarray:
        """extract events from a channel and return them in reference time."""
        assert stream_name in self.target_streams, f"Stream '{stream_name}' not found"

        stream = self.target_streams[stream_name]

        print(f"\nAligning channel {channel_number} from '{stream_name}'...")

        # SpikeGLX convention: digital/sync words always in last channel
        dig_array = extract_digital_channel(stream["data"], -1)

        event_onsets = extract_pulses_with_duration(
            dig_array[channel_number], stream["sample_rate"], **pulse_params
        )

        print(f"{_TEAL}  Extracted {len(event_onsets)} events{_RESET}")

        aligned_onsets = self._correct_times(event_onsets, stream_name)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{stream_name}_ch{channel_number}_aligned.txt"
            np.savetxt(output_file, aligned_onsets, fmt="%.6f")
            print(f"  Saved to {output_file}")

        return aligned_onsets

    def align_channels(self, stream_name: str, channels: List[int],
                      pulse_params_list: List[dict],
                      output_dir: Optional[Path] = None) -> Dict[int, np.ndarray]:
        """extract and align multiple channels."""
        assert len(channels) == len(pulse_params_list), \
            "channels and pulse_params_list must have same length"

        results = {}
        for ch, params in zip(channels, pulse_params_list):
            results[ch] = self.align_channel(stream_name, ch, params, output_dir)

        return results
