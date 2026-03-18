"""
Data stream alignment using sync pulses.

Aligns multi-device recordings (e.g., NIDQ + Neuropixels) by detecting shared
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




def read_meta(bin_path: Path) -> dict:
    """Parse SGLX meta file into dictionary. Works for .bin and .cbin (replaces last suffix)."""
    meta_path = Path(bin_path).with_suffix(".meta")
    meta_dict = {}
    with open(meta_path) as meta_file:
        for line in meta_file:
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                
                # parse numeric values
                if val and re.fullmatch(r"[0-9,.]*", val) and val.count(".") < 2:
                    parsed = [float(v) for v in val.split(",")]
                    val = parsed[0] if len(parsed) == 1 else parsed
                
                meta_dict[key] = val
    return meta_dict


def read_channel_from_cbin(cbin_path: Path, channel_idx: int) -> np.ndarray:
    """
    Read a single channel from a .cbin file entirely in memory (no .bin written to disk).

    Each chunk is decompressed, the requested channel is extracted, and the rest is discarded.
    """
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
        chunk = r.read_chunk(chunk_idx, chunk_start, chunk_length)  # (n_samples_chunk, n_channels)
        chunks.append(chunk[:, channel_idx].copy())
    r.close()
    return np.concatenate(chunks)


def make_memmap(bin_path: Path, meta: dict) -> np.memmap:
    """Memory-map binary data file."""
    n_chan = int(meta["nSavedChans"])
    n_samp = int(int(meta["fileSizeBytes"]) / (2 * n_chan))
    
    return np.memmap(
        bin_path,
        dtype=np.int16,
        mode="r",
        shape=(n_chan, n_samp),
        order="F"
    )


def unpack_bits(channel_data: np.ndarray) -> np.ndarray:
    """Unpack all 16 bits from a 1D int16 channel array. Returns (16, n_samples) uint8."""
    dig = np.zeros((16, len(channel_data)), dtype=np.uint8)
    for bit in range(16):
        dig[bit, :] = (channel_data >> bit) & 0x01
    return dig


def extract_digital_channel(raw_data: np.memmap, channel_idx: int) -> np.ndarray:
    """Extract and unpack a digital channel from a 2D memmap. Returns (16, n_samples) uint8."""
    if channel_idx < 0:
        channel_idx = raw_data.shape[0] + channel_idx
    return unpack_bits(raw_data[channel_idx, :])


def merge_nearby_pulses(onset_samples: np.ndarray, offset_samples: np.ndarray,
                       sample_rate: float, max_gap_ms: float = 1.0,
                       target_duration_ms: float = 28.0, 
                       tolerance_ms: float = 5.0) -> tuple:
    """Merge consecutive pulses separated by short gaps."""
    if len(onset_samples) == 0:
        return onset_samples, offset_samples
    
    max_gap_samples = int((max_gap_ms / 1000) * sample_rate)
    min_combined_dur = target_duration_ms - tolerance_ms
    max_combined_dur = target_duration_ms + tolerance_ms
    
    merged_onsets = []
    merged_offsets = []
    
    i = 0
    while i < len(onset_samples):
        current_onset = onset_samples[i]
        current_offset = offset_samples[i]
        
        if i + 1 < len(onset_samples):
            next_onset = onset_samples[i + 1]
            gap = next_onset - current_offset
            
            if gap <= max_gap_samples:
                next_offset = offset_samples[i + 1]
                combined_duration_samples = next_offset - current_onset
                combined_duration_ms = (combined_duration_samples / sample_rate) * 1000
                
                if min_combined_dur <= combined_duration_ms <= max_combined_dur:
                    merged_onsets.append(current_onset)
                    merged_offsets.append(next_offset)
                    i += 2
                    continue
        
        merged_onsets.append(current_onset)
        merged_offsets.append(current_offset)
        i += 1
    
    return np.array(merged_onsets), np.array(merged_offsets)


def extract_pulses_with_duration(pulse_data: np.ndarray, sample_rate: float,
                                 target_duration_ms: float = None,
                                 tolerance_ms: float = 2.0,
                                 merge_gap_ms: float = 0.0) -> np.ndarray:
    """Extract pulse onset times, optionally filtering by duration.

    If target_duration_ms is None, every rising edge is returned (no duration filter).
    """
    data = pulse_data.astype(np.int8)
    diffs = np.diff(data, prepend=0, append=0)

    onset_samples = np.where(diffs == 1)[0]
    offset_samples = np.where(diffs == -1)[0]

    assert len(onset_samples) == len(offset_samples), "Onset/offset mismatch"

    if target_duration_ms is None:
        return onset_samples / sample_rate

    # Merge nearby pulses if requested
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
        print(f"  Duration filter: rejected {n_rejected} pulse(s) outside "
              f"{min_dur:.1f}–{max_dur:.1f} ms window:")
        for t, d in zip(rejected_times, rejected_durs):
            print(f"    t={t:.3f} s  dur={d:.2f} ms")

    return onset_samples[mask] / sample_rate


class DataStreamAligner:
    """
    Align multi-device recordings using sync pulses.
    
    Corrects temporal drift between a reference stream (e.g., Neuropixels AP)
    and target streams (e.g., NIDQ) using shared sync pulses.
    """
    
    def __init__(self, reference_file: Path, reference_sync_channel: int,
                 sync_params: dict, reference_sync_bit: int = 0,
                 cache_dir: Optional[Path] = None):
        """
        Initialize aligner with reference stream.

        Parameters
        ----------
        reference_file : Path
            Path to reference .bin or .cbin file (e.g., .ap.bin)
        reference_sync_channel : int
            Channel index for sync pulses (negative indexing supported)
        sync_params : dict
            Pulse extraction parameters:
            - target_duration_ms: expected pulse duration
            - tolerance_ms: duration tolerance
            - merge_gap_ms: max gap for merging split pulses
        reference_sync_bit : int
            Bit number within sync channel (0-15). For AP files, use 6.
        """
        self.reference_file = Path(reference_file)
        self.reference_sync_channel = reference_sync_channel
        self.reference_sync_bit = reference_sync_bit

        # pop alignment-control keys before passing sync_params to extract_pulses_with_duration
        sync_params = dict(sync_params)
        self.max_trim = sync_params.pop("max_trim", 1)
        self.check_sync_ipi = sync_params.pop("check_sync_ipi", True)
        self.sync_params = sync_params

        # Load reference metadata (.cbin has a .ap.meta sidecar with the same stem)
        self.reference_meta = read_meta(self.reference_file)
        assert self.reference_meta, f"Meta file not found for {self.reference_file}"

        self.reference_sample_rate = float(self.reference_meta["imSampRate"])

        # Extract reference sync pulses — read only the sync channel, no .bin written to disk
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
        self._ref_sync_line = ref_dig[reference_sync_bit]  # kept for truncation check

        print(f"  Using bit {reference_sync_bit} for sync")
        self.reference_sync_onsets = extract_pulses_with_duration(
            ref_dig[reference_sync_bit], self.reference_sample_rate, **self.sync_params
        )
        
        print(f"  Found {len(self.reference_sync_onsets)} sync pulses")
        if self.check_sync_ipi:
            self.reference_sync_onsets = self._check_ipi(self.reference_sync_onsets, "reference")

        # Target streams storage
        self.target_streams: Dict[str, dict] = {}
    
    def _check_ipi(self, onsets: np.ndarray, stream_name: str) -> np.ndarray:
        """
        Check sync pulse inter-pulse intervals for irregularities.

        Warns in red for short IPIs (possible spurious pulse).
        Fills long IPI gaps with synthetic onsets and warns in orange.
        Returns (possibly extended) onset array.
        """
        if len(onsets) < 2:
            return onsets

        ORANGE = "\033[1;33m"
        RED = "\033[1;31m"
        RESET = "\033[0m"

        ipis = np.diff(onsets)
        median_ipi = float(np.median(ipis))

        # warn on short IPIs (< 50% of expected) — cannot auto-fix
        short_idx = np.where(ipis < 0.5 * median_ipi)[0]
        for i in short_idx:
            print(
                f"{RED}  !!! WARNING: SHORT IPI in '{stream_name}' between pulses "
                f"{i} and {i+1}: {ipis[i]*1000:.1f} ms "
                f"(expected ~{median_ipi*1000:.0f} ms) — possible spurious pulse !!!{RESET}"
            )

        # fill long IPIs (> 150% of expected) with synthetic pulses
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
                f"{ORANGE}  !!! WARNING: MISSING SYNC PULSE in '{stream_name}' after pulse {i} "
                f"(t={onsets[i]:.3f} s): IPI = {ipis[i]*1000:.1f} ms, "
                f"expected ~{median_ipi*1000:.0f} ms. "
                f"Inserted {n_missing} synthetic pulse(s). !!!{RESET}"
            )
            offset += n_missing

        return np.array(corrected)

    def add_target_stream(self, target_file: Path, target_sync_channel: int,
                         stream_name: str, target_sync_bit: int = 0):
        """
        Add target stream to align.
        
        Parameters
        ----------
        target_file : Path
            Path to target .bin or .cbin file
        target_sync_channel : int
            Channel index for sync pulses
        stream_name : str
            Identifier for this stream
        target_sync_bit : int
            Bit number within sync channel (0-15)
        """
        target_file = Path(target_file)

        # Load target metadata
        target_meta = read_meta(target_file)
        assert target_meta, f"Meta file not found for {target_file}"

        target_sample_rate = float(target_meta.get("niSampRate", target_meta.get("imSampRate")))

        # Extract target sync pulses
        print(f"\nExtracting sync from target '{stream_name}': {target_file.name}")
        if target_file.suffix == ".cbin":
            sync_channel_data = read_channel_from_cbin(target_file, target_sync_channel)
            target_dig = unpack_bits(sync_channel_data)
            target_data = None  # cbin targets not supported for align_channel; load lazily if needed
        else:
            target_data = make_memmap(target_file, target_meta)
            target_dig = extract_digital_channel(target_data, target_sync_channel)
        
        print(f"  Using bit {target_sync_bit} for sync")
        target_sync_onsets = extract_pulses_with_duration(
            target_dig[target_sync_bit], target_sample_rate, **self.sync_params
        )

        print(f"  Found {len(target_sync_onsets)} sync pulses")
        if self.check_sync_ipi:
            target_sync_onsets = self._check_ipi(target_sync_onsets, stream_name)

        # Validate sync pulse counts match; trim trailing extra pulses if mismatch is small
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

                # Determine which end has the extra pulse by comparing first pulse times.
                # If the first pulses are already aligned, the extra pulse is at the end.
                # If they differ by ~one period, the extra pulse is at the beginning.
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

                # Check whether the sync line was high at the trimmed sample (confirms truncation)
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

                ORANGE = "\033[1;33m"
                RESET = "\033[0m"
                print(f"{ORANGE}  !!! WARNING: TRIMMED {diff} SYNC PULSE(S) FROM {trim_end.upper()} OF {longer.upper()} !!!")
                print(f"  Counts now match: {n_keep}{RESET}")
                print(truncation_msg)
            else:
                raise ValueError(
                    f"Sync pulse count mismatch after IPI gap-filling!\n"
                    f"  Reference: {n_ref} pulses, last at {ref_last:.3f} s, span {ref_dur:.1f} s\n"
                    f"  Target:    {n_tgt} pulses, last at {tgt_last:.3f} s, span {tgt_dur:.1f} s\n"
                    f"  Difference: {diff} pulse(s) — too large to auto-trim (max_trim={self.max_trim})\n"
                    f"  Check for spurious pulses (short-IPI warnings above) or multiple missing pulses."
                )
        
        # Calculate drift
        drift = self.reference_sync_onsets - target_sync_onsets
        max_drift = np.max(np.abs(drift))
        
        print(f"  Max drift: {max_drift*1000:.2f} ms")
        if max_drift > 0.1:  # 100ms
            print(f"  WARNING: Drift exceeds 100ms threshold!")
        
        # Store stream info
        self.target_streams[stream_name] = {
            "file": target_file,
            "meta": target_meta,
            "sample_rate": target_sample_rate,
            "sync_onsets": target_sync_onsets,
            "data": target_data,
        }
    
    def _correct_times(self, target_times: np.ndarray, stream_name: str) -> np.ndarray:
        """
        Apply nearest-sync correction to event times.
        
        Uses vectorized nearest-neighbor lookup for speed.
        """
        stream = self.target_streams[stream_name]
        target_sync = stream["sync_onsets"]
        
        # Find nearest sync pulse for each event (vectorized)
        indices = np.searchsorted(target_sync, target_times)
        indices = np.clip(indices, 0, len(target_sync) - 1)
        
        # Calculate shifts
        shifts = self.reference_sync_onsets[indices] - target_sync[indices]
        
        # Apply correction
        return target_times + shifts
    
    def align_channel(self, stream_name: str, channel_number: int,
                     pulse_params: dict, output_dir: Optional[Path] = None) -> np.ndarray:
        """
        Extract and align events from a channel.
        
        Parameters
        ----------
        stream_name : str
            Target stream identifier
        channel_number : int
            Channel to extract events from
        pulse_params : dict
            Pulse extraction parameters (duration, tolerance, merge_gap)
        output_dir : Path, optional
            Directory to save aligned times
        
        Returns
        -------
        np.ndarray
            Aligned event times (in reference time)
        """
        assert stream_name in self.target_streams, f"Stream '{stream_name}' not found"
        
        stream = self.target_streams[stream_name]
        
        print(f"\nAligning channel {channel_number} from '{stream_name}'...")
        
        # Extract digital channel (SpikeGLX convention: digital/sync words always in last channel)
        dig_array = extract_digital_channel(stream["data"], -1)
        
        # Extract events from specified bit within the digital word
        event_onsets = extract_pulses_with_duration(
            dig_array[channel_number], stream["sample_rate"], **pulse_params
        )
        
        print(f"  Extracted {len(event_onsets)} events")
        
        # Correct times
        aligned_onsets = self._correct_times(event_onsets, stream_name)
        
        # Save if output_dir provided
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
        """
        Extract and align multiple channels (batch processing).
        
        Parameters
        ----------
        stream_name : str
            Target stream identifier
        channels : list of int
            Channel numbers to extract
        pulse_params_list : list of dict
            Pulse parameters for each channel
        output_dir : Path, optional
            Directory to save aligned times
        
        Returns
        -------
        dict
            {channel_number: aligned_times}
        """
        assert len(channels) == len(pulse_params_list), \
            "channels and pulse_params_list must have same length"
        
        results = {}
        for ch, params in zip(channels, pulse_params_list):
            results[ch] = self.align_channel(stream_name, ch, params, output_dir)
        
        return results
