# Workload Inference — CLI Script Reference

All scripts are installed as console entry points by `uv sync` (or `pip install -e .`).
Run them from anywhere once the package is installed, or with `uv run <script>` from
the `services/workload_inference/` directory.

The default data root is `services/workload_inference/data/experiments/`.

---

## `workload_inference` — Live Experiment Runner

Starts the real-time workload inference GUI for a running experiment.
Connects to the drone swarm and eye tracker via shared memory / ZMQ, runs the
cognitive workload classifier, and manages the experiment state machine.

```
workload_inference [--experiment {nback,gates}]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--experiment` | `nback` \| `gates` | `nback` | Which experiment protocol to run. `nback` runs the N-back cognitive task; `gates` runs the gate-racing task. |

**Examples**
```bash
workload_inference                      # N-back experiment (default)
workload_inference --experiment gates   # Gate racing experiment
```

> **Requires** a running drone simulator and eye tracker connected via shared memory.

---

## `visualize_task` — Offline Replay Viewer

Replays a recorded trial folder and shows a side-by-side visualisation of gaze data,
drone positions, and (optionally) real-time workload inference computed on the fly
from the recorded gaze stream.

```
visualize_task [trial_folder] [--model MODEL]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `trial_folder` | path (positional, optional) | `data/experiments/experiment_nback/ALH0/FlyingPractice` | Path to a trial folder containing `gaze_data.csv` and `drone_data.csv`. Can be relative (resolved against the data root) or absolute. |
| `--model`, `-m` | path | none | Path to a trained model file. When provided, workload inference runs live during replay and the workload display widget is shown. Without it, only gaze and drone data are visualised. |

**Examples**
```bash
visualize_task                                              # Default trial, no model
visualize_task data/experiments/experiment_nback/TQVW/task_1/trial_2
visualize_task data/experiments/experiment_nback/TQVW/task_1/trial_2 \
    --model data/models/my_model.zip
```

---

## `offline_inference` — Batch Workload Classifier

Scans an experiment folder for trials that have `gaze_data.csv` + `nback_data.csv`,
runs the workload classifier offline, and writes an `inference_data.csv` alongside
each trial. This is the necessary pre-processing step before running `plot_results inference`.

```
offline_inference [--data DIR] [--model MODEL] [--config YAML] [--eye-metrics YAML]
                  [--output DIR] [--overwrite] [--dry-run] [--log-level LEVEL]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data` | path | `data/experiments/` | Root folder to scan. Accepts a single trial folder, a subject folder, or a full experiment folder — all are handled recursively. |
| `--model` | path | auto-detected from `data/models/` | Trained classifier file (`.zip` for TabNet, `.pkl`/`.joblib` for sklearn). The newest file in `data/models/` is used when omitted. |
| `--config` | path | built-in defaults | `InferenceSettings` YAML file controlling window size, step parameters, and Schmitt filter thresholds. |
| `--eye-metrics` | path | `data/eye_metrics.yml` | Eye metrics preprocessing config (blink detection, saccade thresholds, etc.). |
| `--output` | path | alongside each `gaze_data.csv` | If provided, writes `inference_data.csv` files into this directory, mirroring the source folder structure instead of writing in-place. |
| `--overwrite` | flag | false | Re-process trials that already have an `inference_data.csv`. Without this flag, already-processed trials are skipped. |
| `--dry-run` | flag | false | Print which folders would be processed without actually running inference. Useful for checking scope. |
| `--log-level` | `DEBUG`\|`INFO`\|`WARNING`\|`ERROR` | `WARNING` | Logging verbosity. Use `INFO` to see per-trial progress, `DEBUG` for detailed internal state. |

**Examples**
```bash
# Process all unprocessed trials in an experiment
offline_inference --data data/experiments/experiment_nback

# Process a single subject, overwriting existing results
offline_inference --data data/experiments/experiment_nback/TQVW --overwrite

# Preview what would be processed without running anything
offline_inference --data data/experiments/experiment_nback --dry-run

# Use a specific model and show INFO-level progress
offline_inference --data data/experiments/experiment_nback \
    --model data/models/tabnet_v3.zip \
    --log-level INFO
```

---

## `plot_results` — Inference & Racing Analysis Plots

Generates publication-quality plots from processed experiment data.
Two modes exist: `inference` (N-back workload accuracy) and `racing` (gate racing
trajectory and adaptation).

```
plot_results {inference,racing} [--data DIR] [--output DIR] [--show] [--cwl {0,1,2}]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `result_type` | `inference` \| `racing` | *(required)* | Which analysis to run (see modes below). |
| `--data` | path | `data/experiments/` | Experiment folder (trial / subject / experiment mode — auto-detected). |
| `--output` | path | `data/results/` | Directory where PNG files are saved. Created if absent. |
| `--show` | flag | false | Open an interactive matplotlib window after saving. |
| `--cwl` | `0` \| `1` \| `2` | none | *(inference mode only)* Overlay a trajectory coloured by CWL level for the task corresponding to this cognitive load level (0 = Low, 1 = Medium, 2 = High). Task assignment is resolved automatically per subject. |

### Mode: `inference`

Requires `inference_data.csv` in the target folder(s) — run `offline_inference` first.

Produces:
- **`inference_time_series.png`** — workload state over time (raw vs. filtered), ground-truth N-back level, and rolling accuracy.
- **`inference_accuracy_summary.png`** — overall accuracy and per-class recall bars.
- *(subject/experiment mode)* **`inference_subject_summary.png`** — per-task and per-CWL-level accuracy bars for each subject.
- *(with `--cwl`)* **`inference_trajectory_cwlN.png`** — drone trajectory coloured by workload prediction for the chosen CWL level.

### Mode: `racing`

Produces:
- **`racing_course_analysis.png`** — 3D trajectory with gate overlays, CWL over time, and raw control inputs.
- **`racing_adaptation_performance.png`** — normalised adaptation parameters over time, split times per gate, and a performance summary.

**Examples**
```bash
# Inference plots for a single subject
plot_results inference --data data/experiments/experiment_nback/TQVW --show

# Inference plots for all subjects in an experiment
plot_results inference --data data/experiments/experiment_nback --output data/results/nback

# Inference with CWL trajectory overlay for the High workload condition
plot_results inference --data data/experiments/experiment_nback/TQVW --cwl 2

# Racing plots for a single trial
plot_results racing --data data/experiments/experiment_racing_dryrun/ERK0/task_0/trial_1 --show

# Racing plots for a full subject
plot_results racing --data data/experiments/experiment_racing_dryrun/ERK0
```

---

## `plot_command_limits` — CWL Step Statistics & Flight Profile Definition

Reads all `command_data.csv` files in an experiment folder and computes **global
and per-subject statistics** on the CWL step number (`cwl_current_step`) used
throughout the experiments.

The intended workflow is:
1. Run this script to see the average and median step numbers achieved by subjects.
2. Decide which value (mean or median) represents the "ideal" step for your
   control-group profile.
3. Manually compute the corresponding limit values using your system's step-to-limit
   mapping.
4. Update the `FLIGHT_PROFILE_LIMITS` dict at the top of the script with those
   values.
5. Use those constants in future control-group experiments (no real-time CWL
   adaptation).

```
plot_command_limits [--data DIR]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data` | path | `data/experiments/` | Experiment folder (trial / subject / experiment mode — auto-detected). Must contain `command_data.csv` files with `cwl_current_step` and `cwl_total_steps` columns. |

Prints to stdout:
- Global statistics: mean step, median step, min/max observed, total steps.
- Per-subject statistics: mean/median/range, number of data rows, number of trials.
- Instructions for manually defining the flight profile limits.
- (If defined) the current `FLIGHT_PROFILE_LIMITS` constants.

**Examples**
```bash
# Analyse all subjects in a racing experiment
plot_command_limits --data data/experiments/experiment_racing_dryrun

# Analyse a single subject
plot_command_limits --data data/experiments/experiment_racing_dryrun/ERK0
```

---

## `generate_data` — Fake Data Generator (Development Only)

Generates synthetic `gaze_data.csv` and `drone_data.csv` files into
`data/experiments/test_experiment/` for testing the pipeline without real hardware.

```
generate_data [-t SECONDS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `-t`, `--time` | int | `5` | Duration of data generation in seconds. Stop early with Ctrl+C. |

**Examples**
```bash
generate_data           # Generate 5 seconds of test data
generate_data -t 30     # Generate 30 seconds of test data
```

> **Note:** This is a development utility. The generated data can be replayed with
> `visualize_task` or fed into `offline_inference` for pipeline testing.

---

## Typical Workflows

### Run an N-back experiment and analyse results

```bash
# 1. Run the live experiment (records CSVs automatically)
workload_inference --experiment nback

# 2. Run offline inference on the recorded data
offline_inference --data data/experiments/experiment_nback --log-level INFO

# 3. Generate accuracy plots
plot_results inference --data data/experiments/experiment_nback --show
```

### Run a gate racing experiment and analyse the flight profile

```bash
# 1. Run the live experiment
workload_inference --experiment gates

# 2. Generate racing analysis plots (no inference needed)
plot_results racing --data data/experiments/experiment_racing_dryrun --show

# 3. Analyse limit distributions to derive a recommended flight profile
plot_command_limits --data data/experiments/experiment_racing_dryrun --show
```

### Test the pipeline without hardware

```bash
# 1. Generate synthetic sensor data
generate_data -t 60

# 2. Replay and inspect it visually
visualize_task data/experiments/test_experiment
```

---

## Data Folder Structure

```
services/workload_inference/data/
├── experiments/
│   ├── experiment_nback/
│   │   └── <SUBJ>/              # 4-char subject ID (e.g. TQVW)
│   │       ├── FlyingPractice/
│   │       │   ├── gaze_data.csv
│   │       │   ├── drone_data.csv
│   │       │   └── nback_data.csv
│   │       └── task_N/
│   │           └── trial_M/
│   │               ├── gaze_data.csv
│   │               ├── drone_data.csv
│   │               ├── nback_data.csv
│   │               └── inference_data.csv   ← written by offline_inference
│   └── experiment_racing_dryrun/
│       └── <SUBJ>/
│           └── task_N/
│               └── trial_M/
│                   ├── gaze_data.csv
│                   ├── drone_data.csv
│                   └── command_data.csv     ← contains limit columns
├── models/                                  ← trained classifiers
└── results/                                 ← plot outputs
```
