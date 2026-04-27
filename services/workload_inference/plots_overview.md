# Racing Plots — Overview & Output Reference

Quick reference for the plots produced by `plot_results racing` after the recent
additions. Mode is auto-detected from the path passed via `--data`:

| Path shape | Mode | Detection rule |
|---|---|---|
| `…/trial_N/` (contains `gate_layout.csv`) | **trial** | file exists at root |
| `…/<SUBJ>/` (4-char `[A-Z0-9]{4}`) | **subject** | folder name matches subject regex |
| any parent containing subject folders | **experiment** | otherwise |

```bash
plot_results racing --data data/experiments/experiment_racing_gate/BEN0/trial_1
plot_results racing --data data/experiments/experiment_racing_gate/BEN0
plot_results racing --data data/experiments/experiment_racing_gate
```

Outputs land in `--output` (default `data/results/`).

---

## Trial mode

### `racing_course_analysis.png` *(unchanged)*
Top-down course overview with gate bars, centroid trajectory coloured by CWL
(or by adaptation step when `--type adaptive`), CWL probability stack, profile
fill, and per-gate drone-pass bars — all sharing the Z-axis.

### `racing_adaptation_performance.png` *(updated)*
- **Top (2×3 grid)** — per-parameter flight profile limit over time, computed
  as `vmin + (vmax − vmin) · cwl_current_step/(N−1)` for `max_pitch`,
  `max_roll`, `max_yaw_rate`, `max_speed`, `max_altitude_rate`, `max_alpha`.
  Three filled bands per panel: green (always-on minimum), blue (range
  unlocked by current CWL step), red (locked headroom). Min/max lines dashed.
- **Bottom-left** — gate split times bar chart (unchanged).
- **Bottom-right** — performance summary text. *Now reports* **Completion
  time** as `{m}m {s}s` instead of an elapsed-seconds float.

The flight-profile constants live in `plot_results.py` as
`_FLIGHT_PROFILE_LIMITS` / `_FLIGHT_PROFILE_MIN_LIMITS` and mirror the
authoritative values in `plot_command_limits.py`.

---

## Subject mode (new)

### `racing_subject_<ID>_completion_times.png`
Horizontal stacked bars, one per trial.

- Segments split at every gate passage.
- **Hue = difficulty**: green segments are easy, red are hard (taken from
  the destination gate's `is_hard`).
- **Two shades per hue alternate** along the bar so adjacent segment
  boundaries stay visible without losing the difficulty cue.
- Per-segment split annotated inside each segment when wide enough (`X.Xs`).
- Total completion annotated at the right end as `{m}m {s}s`.

### `racing_subject_<ID>_segment_cwl.png`
Per-course-segment CWL profile across all of a subject's trials.

- **Y-axis** groups course segments (`Easy 1`, `Hard 1`, `Easy 2`, …),
  named via per-difficulty counters in passage order.
- Inside each segment band, **one stripe per trial** (`T1, T2, …`).
- Each stripe is coloured per sample by `filtered_state`
  (Low / Medium / High).
- **Time inside a segment is normalised to `[0, 1]`**, so a slow trial
  and a fast trial line up horizontally — you can read CWL at "30 % through
  Hard 2" across all trials at the same x-coordinate.
- Soft background band (green/red) reinforces difficulty.

---

## Experiment mode (new)

Subjects are auto-classified by inspecting `cwl_current_step` across all
their trials:

- varies → `adaptive` (blue)
- constant → `non_adaptive` / control (orange)
- no command data → `unknown` (gray)

The classification is printed to stdout so it can be sanity-checked.

### `racing_experiment_performance.png`
2×2 grid of per-subject box plots, subjects ordered adaptive → control → unknown.

| Panel | Metric | Better |
|---|---|---|
| top-left | Completion time (s) | ↓ |
| top-right | Min drones alive (/9) | ↑ |
| bottom-left | Σ drone-gate misses (alive but outside) | ↓ |
| bottom-right | **Composite penalty** = `completion + 30·dead + 5·miss` | ↓ |

Each box shows that subject's trial distribution; per-trial dots are jittered
overlaid; group means are dashed horizontal lines annotated with μ; a vertical
dotted line separates the two groups.

The penalty constants are at the top of the racing section in
`plot_results.py`:

```python
RACING_DEAD_PENALTY_S = 30.0
RACING_MISS_PENALTY_S = 5.0
```

Tuning rationale: a dead drone is an unrecoverable loss → heavy fixed cost
(~½ a typical lap time). A miss is per-event and should *accumulate* — five
seconds compounds quickly for a chronic straggler.

### `racing_experiment_cwl_adaptation.png`
- **Left** — stacked horizontal bars: each subject's time fraction in
  Low / Medium / High CWL, sorted by group, labelled `<SID> [Adaptive]` etc.
- **Right** — box plot of `cwl_current_step` per **adaptive** subject only
  (control subjects have constant step by construction). Grand mean across
  all adaptive trials shown as a dashed black line; min/max step references
  shown as dotted gray lines.

Stdout also prints a `GROUP SUMMARY` block (mean ± std for completion,
min_alive, missed, dead, penalty) that's useful even before the control
group is complete — gives you the adaptive baseline to compare against
once control trials land.

---

## Suggested variants & next steps

### Strengthening the adaptive vs control comparison
Once the four missing control subjects are recorded:

1. **Effect size + bootstrap CI on the penalty score.** With small `n`
   (≈5 per group), report Cohen's *d* and a 95 % bootstrap CI on
   `(control − adaptive) penalty`. Add as a sidecar text annotation on the
   penalty panel — more honest than a t-test p-value at this sample size.
2. **Paired design if feasible.** If each subject can run both conditions
   (counterbalanced), within-subject Δ collapses individual skill variance
   and is far more powerful than between-subject boxes. A "slope chart"
   (one line per subject, control endpoint → adaptive endpoint, coloured
   by direction) would show the per-subject benefit at a glance.
3. **Per-segment performance breakdown.** The `Hard N` segments are likely
   where adaptation pays off. Box plot of split times per *segment type*
   (Easy vs Hard) split by group would isolate that effect; a uniform
   speed-up everywhere would be less interesting than a targeted hard-segment
   improvement.

### Validating the adaptation mechanism (independent of outcomes)
4. **CWL → step coupling scatter.** Per adaptive trial, plot
   `mean(filtered_state)` vs `mean(cwl_current_step)` — confirms the
   system is actually responding to inferred workload before claiming the
   *outcomes* are due to adaptation.
5. **Step-response latency.** Cross-correlate `filtered_state` and
   `cwl_current_step` to estimate the lag between a CWL change and the
   resulting profile change. Useful tuning knob, and a flat correlation
   would flag a broken pipeline.

### Per-subject / per-trial richness
6. **Confusion matrix per trial** (`nback_level` vs `filtered_state`) —
   shows whether errors cluster at specific transitions (e.g. always
   collapsing High → Medium under fatigue). Already raised earlier;
   straightforward to add.
7. **Rolling completion-time across trials.** One panel per subject,
   trial index on x, completion on y — reveals learning curves and lets
   you decide whether to drop the first trial as a warmup.
8. **Spline-based segment projection.** The current segment plot uses
   *time*-normalised x. A second variant using *arc-length* on the spline
   (already implemented for inference plots via `_project_to_arc`) would
   align trials by physical track position rather than time, making
   "stuck at the chicane" patterns more visible.

### Style improvements
9. **Consistent group palette across all figures.** Currently
   `GROUP_COLORS` is used in the experiment plots only — extending it to
   the subject-mode plots (e.g. tinting the trial bars in
   `racing_subject_..._completion_times.png` by the subject's group)
   would let mixed-group reports stay visually coherent.
10. **Optional per-group facet rows** in the experiment box plots for
    `n ≥ 8` per group — once data piles up, side-by-side panels per
    group become easier to read than the mixed-axis approach.
