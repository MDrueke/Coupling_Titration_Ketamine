"""
Extract pulse onset times from NIDQ digital channels.

Reads the nidq binary file, auto-detects which digital line contains the
optogenetic pulse signal, and extracts pulse onset times.
"""

from pathlib import Path
import numpy as np
import tomllib


# =============================================================================
# PULSE EXTRACTION PARAMETERS
# =============================================================================

DEBUG = False  # Set to True for verbose output and saving raw pulse times

# Sync pulse parameters (Line 0)
SYNC_PARAMS = {
    "target_duration_ms": 500.0,
    "tolerance_ms": 5.0,
    "merge_gap_ms": 1.5,
}

# Stimulus pulse parameters (Line 7)
STIM_PARAMS = {
    "target_duration_ms": 28.0,
    "tolerance_ms": 2.0,
    "merge_gap_ms": 1.5,
}

EXPECTED_STIM_COUNT = 600  # 601 lines in CSV - 1 header


# =============================================================================
# NIDQ READING FUNCTIONS (adapted from readSGLX.py)
# =============================================================================

def read_meta(bin_path: Path) -> dict:
    """Parse SGLX meta file into dictionary."""
    meta_path = bin_path.with_suffix(".meta")
    meta_dict = {}
    with meta_path.open() as f:
        for line in f.read().splitlines():
            key, value = line.split("=", 1)
            if key.startswith("~"):
                key = key[1:]
            meta_dict[key] = value
    return meta_dict


def get_sample_rate(meta: dict) -> float:
    """Get sample rate from meta dictionary."""
    return float(meta["niSampRate"])


def get_channel_counts(meta: dict) -> tuple:
    """Get MN, MA, XA, DW channel counts."""
    counts = meta["snsMnMaXaDw"].split(",")
    return int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3])


def make_memmap(bin_path: Path, meta: dict) -> np.memmap:
    """Memory-map the binary file."""
    n_chan = int(meta["nSavedChans"])
    n_samples = int(meta["fileSizeBytes"]) // (2 * n_chan)
    print(f"nChan: {n_chan}, nSamples: {n_samples}")
    return np.memmap(bin_path, dtype="int16", mode="r",
                     shape=(n_chan, n_samples), order="F")


def extract_digital(raw_data: np.memmap, first_samp: int, last_samp: int,
                    dw_req: int, line_list: list, meta: dict) -> np.ndarray:
    """Extract digital lines from raw data."""
    MN, MA, XA, DW = get_channel_counts(meta)
    dig_ch = MN + MA + XA + dw_req
    
    select_data = np.ascontiguousarray(
        raw_data[dig_ch, first_samp:last_samp + 1], dtype="int16"
    )
    n_samp = last_samp - first_samp + 1
    
    # unpack bits
    bit_data = np.unpackbits(select_data.view(dtype="uint8"))
    bit_data = np.transpose(np.reshape(bit_data, (n_samp, 16)))
    
    n_lines = len(line_list)
    dig_array = np.zeros((n_lines, n_samp), dtype="uint8")
    for i, line in enumerate(line_list):
        byte_n, bit_n = divmod(line, 8)
        targ_i = byte_n * 8 + (7 - bit_n)
        dig_array[i, :] = bit_data[targ_i, :]
    
    return dig_array


# =============================================================================
# PULSE EXTRACTION
# =============================================================================

def load_config(config_path: Path = Path("config.toml")) -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)







def merge_nearby_pulses(onset_samples: np.ndarray, offset_samples: np.ndarray,
                       sample_rate: float, max_gap_ms: float = 1.0,
                       target_duration_ms: float = 28.0, 
                       tolerance_ms: float = 5.0,
                       debug: bool = False) -> tuple:
    """
    Merge consecutive pulses that are separated by a short gap and whose
    combined duration matches the expected pulse length.
    
    Parameters
    ----------
    onset_samples : np.ndarray
        Array of onset sample indices
    offset_samples : np.ndarray
        Array of offset sample indices
    sample_rate : float
        Sampling rate in Hz
    max_gap_ms : float
        Maximum gap between pulses to consider merging (ms)
    target_duration_ms : float
        Expected combined pulse duration (ms)
    tolerance_ms : float
        Tolerance for combined duration matching (ms)
    debug : bool
        If True, print merge information
    
    Returns
    -------
    tuple
        (merged_onset_samples, merged_offset_samples)
    """
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
        
        # Check if next pulse is close enough to merge
        if i + 1 < len(onset_samples):
            next_onset = onset_samples[i + 1]
            gap = next_onset - current_offset
            
            if gap <= max_gap_samples:
                # Calculate combined duration if we merge
                next_offset = offset_samples[i + 1]
                combined_duration_samples = next_offset - current_onset
                combined_duration_ms = (combined_duration_samples / sample_rate) * 1000
                
                # Check if combined duration matches expected
                if min_combined_dur <= combined_duration_ms <= max_combined_dur:
                    # Merge: keep first onset, use second offset
                    merged_onsets.append(current_onset)
                    merged_offsets.append(next_offset)
                    if debug:
                        print(f"    Merged pulses {i} and {i+1}: "
                              f"gap={gap/sample_rate*1000:.3f}ms, "
                              f"combined_dur={combined_duration_ms:.3f}ms")
                    i += 2  # skip both pulses
                    continue
        
        # No merge, keep pulse as-is
        merged_onsets.append(current_onset)
        merged_offsets.append(current_offset)
        i += 1
    
    return np.array(merged_onsets), np.array(merged_offsets)


def extract_pulses_with_duration(pulse_data: np.ndarray, sample_rate: float, 
                               target_duration_ms: float = None, 
                               tolerance_ms: float = 2.0,
                               merge_gap_ms: float = 1.0,
                               debug: bool = False,
                               channel_name: str = "Unknown") -> np.ndarray:
    """
    Extract pulse onset times filtering by pulse duration.
    
    Parameters
    ----------
    pulse_data : np.ndarray
        Digital trace (0s and 1s)
    sample_rate : float
        Sampling rate in Hz
    target_duration_ms : float, optional
        Expected pulse duration in ms. If None, all pulses are returned.
    tolerance_ms : float
        Allowed deviation from target duration in ms.
    merge_gap_ms : float
        Maximum gap between consecutive pulses to attempt merging (ms)
    debug : bool
        If True, enables verbose output and saves raw onset times
    channel_name : str
        Name of the channel being processed (for file naming)
    
    Returns
    -------
    np.ndarray
        Onset times in seconds for valid pulses
    """
    # Force boolean/int8
    data = pulse_data.astype(np.int8)
    
    # Find changes
    diffs = np.diff(data, prepend=0, append=0)
    
    onset_samples = np.where(diffs == 1)[0]
    offset_samples = np.where(diffs == -1)[0]
    
    assert len(onset_samples) == len(offset_samples)
    
    # Save raw onsets if debug mode
    if debug:
        raw_seconds = onset_samples / sample_rate
        filename = f"raw_{channel_name}_onsets.txt"
        np.savetxt(filename, raw_seconds, fmt="%.6f", 
                   header="Raw Pulse Onset Times (Before Filtering) (s)")
        print(f"    Saved raw onsets to '{filename}'")
    
    if debug:
        print(f"    Raw pulses found: {len(onset_samples)}")
    
    # Step 1: Merge nearby short pulses
    if target_duration_ms is not None and merge_gap_ms > 0:
        onset_samples, offset_samples = merge_nearby_pulses(
            onset_samples, offset_samples, sample_rate,
            max_gap_ms=merge_gap_ms,
            target_duration_ms=target_duration_ms,
            tolerance_ms=tolerance_ms,
            debug=debug
        )
        if debug:
            print(f"    After merging: {len(onset_samples)}")
    
    durations_samples = offset_samples - onset_samples
    durations_ms = (durations_samples / sample_rate) * 1000
    
    if target_duration_ms is None:
        return onset_samples / sample_rate
        
    # Step 2: Filter by duration
    min_dur = target_duration_ms - tolerance_ms
    max_dur = target_duration_ms + tolerance_ms
    
    mask = (durations_ms >= min_dur) & (durations_ms <= max_dur)
    
    valid_onsets = onset_samples[mask]
    
    if debug:
        print(f"    Filtered count:   {len(valid_onsets)}")
    
        # Print rejected pulses
        rejected_indices = np.where(~mask)[0]
        if len(rejected_indices) > 0:
            print(f"    Rejected pulses:  {len(rejected_indices)}")
            print(f"\n    {'Pulse Index':<15} {'Onset Time (s)':<20} {'Duration (ms)':<20}")
            print(f"    {'-'*55}")
            for idx in rejected_indices:
                onset_time = onset_samples[idx] / sample_rate
                print(f"    {idx:<15} {onset_time:<20.6f} {durations_ms[idx]:<20.3f}")
            print()
    
    return valid_onsets / sample_rate


def main():
    config = load_config()
    
    nidq_path = Path(config["paths"]["nidqPath"])
    if DEBUG:
        print(f"Loading nidq file: {nidq_path}")
    
    # read meta and create memmap
    meta = read_meta(nidq_path)
    sample_rate = get_sample_rate(meta)
    if DEBUG:
        print(f"Sample rate: {sample_rate} Hz")
    
    raw_data = make_memmap(nidq_path, meta)
    
    # Extract digital word
    n_samples = raw_data.shape[1]
    dw = 0
    d_lines = [0, 7] # Line 0: Sync, Line 7: Stimulus
    
    print("\nExtracting Digital Lines 0 (Sync) and 7 (Stimulus)...")
    dig_array = extract_digital(raw_data, 0, n_samples - 1, dw, d_lines, meta)
    
    # -------------------------------------------------------------------------
    # Process Sync Line (Line 0) - Index 0 in dig_array
    # -------------------------------------------------------------------------
    if DEBUG:
        print("\n--- Processing Sync Channel (Line 0) ---")
        print(f"Parameters: {SYNC_PARAMS}")
    
    sync_onsets = extract_pulses_with_duration(
        dig_array[0], sample_rate, debug=DEBUG, channel_name="sync", **SYNC_PARAMS
    )
    
    if not DEBUG:
        print(f"Sync pulses: {len(sync_onsets)}")
    else:
        print(f"Valid Sync pulses: {len(sync_onsets)}")
        if len(sync_onsets) > 0:
            print(f"  First 5: {sync_onsets[:5]}")
        
    # -------------------------------------------------------------------------
    # Process Stimulus Line (Line 7) - Index 1 in dig_array
    # -------------------------------------------------------------------------
    if DEBUG:
        print("\n--- Processing Stimulus Channel (Line 7) ---")
        print(f"Parameters: {STIM_PARAMS}")
    
    stim_onsets = extract_pulses_with_duration(
        dig_array[1], sample_rate, 
        debug=DEBUG,
        channel_name="stim",
        **STIM_PARAMS
    )
    
    if not DEBUG:
        print(f"Stimulus pulses: {len(stim_onsets)}")
    else:
        print(f"Valid Stimulus pulses: {len(stim_onsets)} (Expected: {EXPECTED_STIM_COUNT})")
        if len(stim_onsets) > 0:
            print(f"  First 5: {stim_onsets[:5]}")
        
    # Always show count mismatch in red
    if len(stim_onsets) != EXPECTED_STIM_COUNT:
        print(f"\033[91m  WARNING: Count mismatch! Found {len(stim_onsets)} vs {EXPECTED_STIM_COUNT}.\033[0m")
        if DEBUG:
            if len(stim_onsets) > EXPECTED_STIM_COUNT:
                 print("  (Found too many. Try reducing tolerance or checking for noise.)")
            else:
                 print("  (Found too few. Try increasing tolerance.)")
    
    # Save to files
    np.savetxt("sync_onsets.txt", sync_onsets, fmt="%.6f", header="Sync Pulse Onset Times (s)")
    np.savetxt("stim_onsets.txt", stim_onsets, fmt="%.6f", header="Stimulus Pulse Onset Times (s)")
    
    if DEBUG:
        print("\nSaved sync onsets to 'sync_onsets.txt'")
        print("Saved stimulus onsets to 'stim_onsets.txt'")
        
    return sync_onsets, stim_onsets


if __name__ == "__main__":
    main()

