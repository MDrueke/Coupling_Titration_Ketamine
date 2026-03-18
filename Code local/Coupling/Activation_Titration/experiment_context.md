# Experiment Context

## Setup
Neuropixels recordings from mouse somatosensory barrel cortex (S1). Animals express Channelrhodopsin in L5b pyramidal neurons via a Sim1-Cre x ChR2 cross. A prism implanted in L1 delivers blue light selectively to the apical tuft dendrites of these L5b cells, enabling compartment-specific optogenetic activation.

## Goal
Characterize a light-intensity-dependent activation curve of L5b pyramidal neurons, and determine whether this curve changes under anesthesia.

## Protocol
- Stimulate with varying light amplitudes (read from `WaveformSequence.csv` in each recording folder).
- Halfway through each recording, the animal is anesthetized.
- Spike responses are recorded on Neuropixels probes; stimulus timing is recorded on NIDQ.

## Pipeline overview
1. **Extract pulses** (`extract_pulses.py`): Detect stimulus and sync pulse onsets from NIDQ digital channels.
2. **Align streams** (`align_datastreams.py`): Correct NIDQ timestamps to AP timebase using shared sync pulses.
3. **Match amplitudes** (`match_amplitudes.py`): Pair aligned pulse times with amplitudes from the waveform CSV (row-order match).
4. **Analyze** (`activation_titration.py`): Load spikes (`recording.py`), compute baseline-normalized firing rates per amplitude, and plot activation curves and PSTHs.
