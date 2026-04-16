# Plan: New Activation Curve Metrics

## Background and motivation

The current metric (mean firing rate in `PULSE_WINDOW` per neuron per amplitude) has two
confounds when comparing awake vs. anesthetized states:

1. **Baseline offset**: anesthetized neurons fire less spontaneously. Even if the absolute
   driven rate is identical across states, ΔFR (driven − baseline) appears higher under
   anesthesia because the baseline is lower. This makes light look more effective under
   anesthesia when it may not be.

2. **Saturation**: at high intensities, all neurons approach their maximum firing rate. A
   neuron that starts at a higher baseline (awake) has less room to increase, so ΔFR
   saturates earlier and looks *lower* than under anesthesia — the opposite of the baseline
   confound. Both effects are real properties of the data, but they make the activation
   curve hard to interpret.

---

## Metric 1: Threshold and slope from the hockey-stick fit

### Status
**Planned, not yet implemented.**

### What it is
The hockey-stick (piecewise linear) fit already computed by `find_activation_threshold`
produces two parameters per session per state:
- **Threshold** (mW/mm²): the intensity at which the population first responds
- **Slope** (Hz per mW/mm² above threshold): how steeply the population is recruited

### Why it is useful
These parameters are baseline-independent (they describe the *shape* of the curve, not its
absolute level) and saturation-resistant (slope is estimated from the rising phase, not the
plateau). They directly capture the gain of the optogenetic drive.

### Implementation plan
1. `find_activation_threshold` already returns both parameters per state. Currently they are
   saved to CSV but not aggregated across sessions.
2. Add collection of threshold and slope into `SessionResult`.
3. Add a pooled plot: paired boxplot or scatter (awake vs. anesthesia) for threshold and
   slope separately, across sessions — one point per session.
4. Statistical test: paired Wilcoxon signed-rank across sessions (already implemented as
   `_wilcoxon`).

### Effort: low — values already computed, just need aggregation and one new plot function.

---

## Metric 2: Response probability

### Status
**Implemented.**

### What it is
For each neuron × amplitude, the fraction of trials in which the neuron fires more than
expected from its baseline. This is baseline-independent and saturation-resistant: a neuron
that fires 20 Hz baseline and 22 Hz driven, and one that fires 0 Hz baseline and 2 Hz
driven, both get the same response probability if the trial-level distribution is similar.

### Parameters

**`RESP_PROB_K = 1.0`**
Controls the per-trial detection threshold:
```
threshold_count = baseline_count + K * baseline_std_count
```
where `baseline_count = baseline_rate * pulse_window_duration` and
`baseline_std_count = sqrt(baseline_rate * pulse_window_duration)` (Poisson approximation).

- `K = 0`: any spike above the expected mean counts → very sensitive, high false positive rate
- `K = 1`: one standard deviation above baseline → moderate sensitivity (default)
- `K = 2`: two standard deviations → conservative, few false positives

For near-zero baseline neurons (e.g. 0.1 Hz, 30 ms window → expected 0.003 spikes),
the threshold is ~0 regardless of K, so even a single spike counts. This is the desired
behaviour and is robust across all K values for these neurons.

**`RESP_PROB_THRESHOLD = 0.5`**
Used only for the per-neuron threshold plot. A neuron's "response threshold" is defined as
the lowest amplitude where its response probability reaches this value. Neurons that never
reach it are excluded from the threshold distribution.

- `0.5`: the neuron responds on more than half of trials — a reasonable definition of
  reliable response
- `0.3`: more inclusive, captures weakly responding neurons
- `0.7`: more conservative, only neurons with strong reliable responses

### How to titrate the parameters

**Titrating `RESP_PROB_K`:**

The right K is the one that best separates signal (light-driven trials) from noise
(spontaneous fluctuations). Two approaches:

1. **Empirical false positive rate**: run the detection on pre-stimulus windows of the same
   duration as `PULSE_WINDOW`. The fraction of "pre-stimulus trials" that exceed the
   threshold is the false positive rate. Choose the smallest K where this rate is
   acceptably low (e.g. < 5%). This can be implemented by computing response probability
   on a shuffled/pre-stimulus window and plotting it as a function of K.

2. **Separation between states at zero intensity**: if you have trials at 0 mW (or very
   low sub-threshold intensities), the response probability should be ~equal between awake
   and anesthetized. A K that gives equal false positive rates across states is well
   calibrated. If awake neurons have higher spontaneous rates, a fixed K will give them
   higher false positives — in that case, K should be increased or a state-specific
   threshold used.

In practice, K = 1 is a reasonable default for Poisson spiking and is consistent with
standard spike detection heuristics. It is unlikely to need tuning unless baseline rates
differ dramatically across states.

**Titrating `RESP_PROB_THRESHOLD`:**

This parameter controls how "reliable" a response must be to count as a threshold crossing.
The right value depends on the scientific question:

- If you want to know at what intensity a neuron *can* be driven (even rarely): use 0.3
- If you want to know at what intensity a neuron *reliably* responds: use 0.5–0.7
- 0.5 is the natural midpoint of the sigmoidal response probability curve and corresponds
  to the EC50 of the intensity-response function — this is the most interpretable and
  commonly used definition

To validate empirically: plot the full response probability curves for individual neurons
(prob vs. amplitude) and inspect whether 0.5 corresponds to a visually meaningful
threshold crossing. Neurons with very shallow curves (never clearly above or below 0.5)
are edge cases that should be excluded regardless of threshold choice.

A sensitivity analysis — running the threshold boxplot at 0.3, 0.5, and 0.7 and checking
whether the awake vs. anesthesia difference is consistent across values — is recommended
before finalising.

### Outputs (per session and pooled)
- `activation_curve/response_probability_curve.pdf` — mean response probability vs.
  intensity, awake vs. anesthesia, with SEM bands
- `activation_curve/response_probability_thresholds.pdf` (pooled only) — boxplot + jitter
  of per-neuron mW thresholds by state, with Mann-Whitney U p-value

### Statistical test note
The threshold boxplot uses **Mann-Whitney U** (not paired Wilcoxon) because neurons that
never reach threshold are excluded per state independently, so the two groups can have
different sizes and cannot be paired. If the analysis is restricted to neurons present and
above-threshold in both states, paired Wilcoxon would be appropriate.
