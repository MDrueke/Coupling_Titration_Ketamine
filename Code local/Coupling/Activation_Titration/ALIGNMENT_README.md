# Data Stream Alignment

Align multi-device recordings using shared sync pulses.

## Quick Start

```python
from align_datastreams import DataStreamAligner

# Initialize with reference stream (e.g., Neuropixels AP)
aligner = DataStreamAligner(
    reference_file="recording.ap.bin",
    reference_sync_channel=-1,  # last channel
    sync_params={
        "target_duration_ms": 500.0,
        "tolerance_ms": 50.0,
        "merge_gap_ms": 0.0
    }
)

# Add target stream (e.g., NIDQ)
aligner.add_target_stream(
    target_file="recording.nidq.bin",
    target_sync_channel=0,  # first digital line
    stream_name="nidq"
)

# Align event channel
stim_times = aligner.align_channel(
    stream_name="nidq",
    channel_number=7,
    pulse_params={
        "target_duration_ms": 28.0,
        "tolerance_ms": 2.0,
        "merge_gap_ms": 1.5
    },
    output_dir="output"
)
# Returns: aligned timestamps (s)
# Saves: output/nidq_ch7_aligned.txt
```

## Batch Processing

```python
# Align multiple channels at once
results = aligner.align_channels(
    stream_name="nidq",
    channels=[7, 8],
    pulse_params_list=[params7, params8],
    output_dir="output"
)
# Returns: {7: times7, 8: times8}
```

## How It Works

1. Extracts sync pulses from reference and target streams
2. Validates pulse counts match (asserts if mismatch)
3. For each event: finds nearest sync pulse in target stream
4. Applies shift from nearest sync to correct event time
5. Warns if max drift > 100ms

## Performance

- **Vectorized**: Corrects thousands of events in milliseconds
- **Memory-efficient**: Uses memory-mapped file access
- **Nearest-sync**: Fast lookup with `np.searchsorted()`
