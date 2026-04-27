"""
Plot and visualize workload inference experiment results.

Three modes are selected automatically based on --data:

* **Trial folder**   (contains inference_data.csv directly)
  → inference_time_series.png only

* **Subject folder** (4-char code, e.g. BEN0 — contains task_N/trial_M sub-folders)
  → inference_time_series.png  (task trials only, concatenated)
  → inference_accuracy_summary.png  (per-task bars + per-CWL-level bars, trial dots)

* **Experiment / root folder** (contains multiple subjects)
  → inference_time_series.png  (all sessions concatenated)
  → inference_accuracy_summary.png  (overall + per-class bars)

When ``--cwl`` is given, trajectory-based CWL plots are produced instead.
The task folder is resolved automatically per subject so randomized task order
across subjects is handled transparently:

* **Subject + cwl**     → trajectory colored by filtered CWL + per-trial accuracy
* **Experiment + cwl**  → aggregate trajectory (mean ± std) + per-subject accuracy

Usage:
    plot_results inference [--show] [--data DIR] [--output DIR] [--cwl {0,1,2}]
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import CheckButtons
from scipy.spatial import cKDTree

_SERVICE_ROOT = Path(__file__).parents[2]
_DEFAULT_DATA = _SERVICE_ROOT / "data" / "experiments"
_DEFAULT_OUTPUT = _SERVICE_ROOT / "data" / "results"
_SPLINE_FILE = _SERVICE_ROOT / "data" / "spline_trajectory.csv"

INFERENCE_FILE_NAME = "inference_data.csv"
DRONE_FILE_NAME = "drone_data.csv"
INFERENCE_SAMPLE_RATE = 60  # samples per second
STATE_LABELS = {0: "Low", 1: "Medium", 2: "High"}
STATE_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}

_TASK_RE = re.compile(r"^task_\d+/")
_SUBJECT_RE = re.compile(r"^[A-Z0-9]{4}$")


# ─────────────────────────────────────────────────────────────────────────────
# Mode detection
# ─────────────────────────────────────────────────────────────────────────────


def _detect_mode(data_dir: Path) -> str:
    """Return 'trial', 'subject', or 'experiment'."""
    if (data_dir / INFERENCE_FILE_NAME).exists():
        return "trial"
    if _SUBJECT_RE.match(data_dir.name):
        return "subject"
    return "experiment"


def _find_task_for_cwl(subject_dir: Path, cwl_level: int) -> str | None:
    """Return the task folder name whose dominant nback_level equals *cwl_level*.

    Iterates task_N sub-folders, reads the first available inference_data.csv
    from any trial, and returns the task name whose majority nback_level matches
    the requested CWL level.  Returns None if no matching task is found.
    """
    task_dirs = sorted(
        d
        for d in subject_dir.iterdir()
        if d.is_dir() and re.match(r"^task_\d+$", d.name)
    )
    if not task_dirs:
        print(f"  {subject_dir.name}: no task folders found.")
        return None

    all_csvs = sorted(subject_dir.rglob(INFERENCE_FILE_NAME))
    if not all_csvs:
        print(
            f"  {subject_dir.name}: no {INFERENCE_FILE_NAME} files found — "
            "run offline_inference first."
        )
        return None

    for task_dir in task_dirs:
        trial_csvs = sorted(task_dir.rglob(INFERENCE_FILE_NAME))
        if not trial_csvs:
            continue
        try:
            df = pd.read_csv(trial_csvs[0], usecols=["nback_level"])
        except Exception:
            continue
        if df.empty:
            continue
        dominant = int(df["nback_level"].mode().iloc[0])
        if dominant == cwl_level:
            return task_dir.name

    cwl_label = STATE_LABELS.get(cwl_level, str(cwl_level))
    print(
        f"  {subject_dir.name}: no task matched CWL level {cwl_label} "
        f"({cwl_level}) — check nback_level values in inference data."
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_inference_data(experiments_dir: Path) -> pd.DataFrame:
    csv_files = sorted(experiments_dir.rglob(INFERENCE_FILE_NAME))
    if not csv_files:
        raise FileNotFoundError(
            f"No {INFERENCE_FILE_NAME} files found under {experiments_dir}"
        )

    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        rel = f.relative_to(experiments_dir)
        df["_source"] = "/".join(rel.parts[:-1])
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data["elapsed_s"] = data.groupby("_source")["timestamp"].transform(
        lambda t: (t - t.iloc[0]) / 1000.0
    )
    return data


def _task_trials_only(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only task_N/trial_M sources (drop FlyingPractice, NBackPractice)."""
    return data[data["_source"].str.match(_TASK_RE)].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _state_step_plot(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, color: str, **kwargs
):
    ax.step(x, y, where="post", label=label, color=color, **kwargs)


def _accuracy_over_time(
    y_true: np.ndarray, y_pred: np.ndarray, window: int = INFERENCE_SAMPLE_RATE
) -> np.ndarray:
    correct = (y_true == y_pred).astype(float)
    return pd.Series(correct).rolling(window, min_periods=1).mean().to_numpy()


def _contiguous_regions(mask: np.ndarray):
    """Yield (start, end) index pairs for contiguous True regions in mask."""
    in_region = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_region:
            start = i
            in_region = True
        elif not val and in_region:
            yield start, i
            in_region = False
    if in_region:
        yield start, len(mask)


def _bar_label(ax: plt.Axes, bars, fmt: str = "{:.0%}", offset: float = 0.02):
    for bar in bars:
        val = bar.get_height()
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=8,
            )


def _hbar_label(ax: plt.Axes, bars, fmt: str = "{:.0%}", offset: float = 0.01):
    for bar in bars:
        val = bar.get_width()
        if val > 0:
            ax.text(
                val + offset,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(val),
                ha="left",
                va="center",
                fontsize=8,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Time series plot  (shared by all modes)
# ─────────────────────────────────────────────────────────────────────────────


def plot_inference_time_series(
    data: pd.DataFrame, ax_states: plt.Axes, ax_rolling: plt.Axes
):
    sources = list(data["_source"].unique())
    offset = 0.0
    x_ticks, x_tick_labels = [], []

    for src in sources:
        seg = data[data["_source"] == src].copy().sort_values("elapsed_s")
        t = seg["elapsed_s"].to_numpy() + offset
        gt = seg["nback_level"].to_numpy()
        raw = seg["raw_state"].to_numpy()
        filt = seg["filtered_state"].to_numpy()

        for level, color in STATE_COLORS.items():
            mask = gt == level
            if not mask.any():
                continue
            for start_i, end_i in _contiguous_regions(mask):
                ax_states.axvspan(
                    t[start_i],
                    t[min(end_i, len(t) - 1)],
                    alpha=0.12,
                    color=color,
                    linewidth=0,
                )

        first = src == sources[0]
        _state_step_plot(
            ax_states,
            t,
            gt,
            label="Ground truth" if first else "_",
            color="#333333",
            linewidth=1.5,
            linestyle="--",
        )
        _state_step_plot(
            ax_states,
            t,
            raw,
            label="Raw inference" if first else "_",
            color="#1976D2",
            linewidth=1.2,
            alpha=0.85,
        )
        _state_step_plot(
            ax_states,
            t,
            filt,
            label="Filtered inference" if first else "_",
            color="#E91E63",
            linewidth=1.5,
        )

        ax_rolling.plot(
            t,
            _accuracy_over_time(gt, raw),
            color="#1976D2",
            linewidth=1.2,
            alpha=0.85,
            label="Raw accuracy" if first else "_",
        )
        ax_rolling.plot(
            t,
            _accuracy_over_time(gt, filt),
            color="#E91E63",
            linewidth=1.5,
            label="Filtered accuracy" if first else "_",
        )

        if len(sources) > 1:
            sep_x = t[-1] + 2.0
            ax_states.axvline(sep_x - 1.0, color="gray", linewidth=0.5, linestyle=":")
            ax_rolling.axvline(sep_x - 1.0, color="gray", linewidth=0.5, linestyle=":")
            x_ticks.append((offset + t[-1]) / 2)
            x_tick_labels.append(src.split("/")[-1])
            offset = sep_x
        else:
            offset = t[-1] + 2.0

    ax_states.set_yticks([0, 1, 2])
    ax_states.set_yticklabels([STATE_LABELS[i] for i in range(3)])
    ax_states.set_ylabel("Workload Level")
    ax_states.set_ylim(-0.3, 2.5)
    ax_states.legend(loc="upper right", fontsize=8)
    ax_states.set_title("Inference Time Series")
    ax_states.grid(axis="y", linestyle=":", alpha=0.4)
    ax_states.set_xticklabels([])

    ax_rolling.set_ylabel(f"Rolling Accuracy\n(window={INFERENCE_SAMPLE_RATE} s)")
    ax_rolling.set_xlabel("Time (s)")
    ax_rolling.set_ylim(-0.05, 1.05)
    ax_rolling.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_rolling.legend(loc="lower right", fontsize=8)
    ax_rolling.grid(axis="y", linestyle=":", alpha=0.4)
    ax_rolling.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    if x_ticks:
        for ax in (ax_states, ax_rolling):
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels, rotation=20, ha="right", fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Subject-level accuracy summary
# ─────────────────────────────────────────────────────────────────────────────


def _build_trial_summary(data: pd.DataFrame) -> pd.DataFrame:
    """One row per source with task, trial, CWL level, raw/filtered accuracy."""
    records = []
    for src, grp in data.groupby("_source"):
        parts = src.split("/")
        task = parts[0] if len(parts) >= 1 else src
        trial = parts[1] if len(parts) >= 2 else "trial_1"
        nback_level = int(grp["nback_level"].mode().iloc[0])
        raw_acc = float((grp["nback_level"] == grp["raw_state"]).mean())
        filt_acc = float((grp["nback_level"] == grp["filtered_state"]).mean())
        records.append(
            {
                "source": src,
                "task": task,
                "trial": trial,
                "nback_level": nback_level,
                "raw_acc": raw_acc,
                "filt_acc": filt_acc,
            }
        )
    return pd.DataFrame(records)


def plot_subject_accuracy_summary(
    data: pd.DataFrame, ax_task: plt.Axes, ax_level: plt.Axes
):
    """Two panels: accuracy per task (with trial dots) + accuracy per CWL level."""
    summary = _build_trial_summary(data)
    width = 0.35
    chance = 1 / 3

    # ── Panel A: per task ────────────────────────────────────────────────────
    tasks = sorted(summary["task"].unique())
    x = np.arange(len(tasks))

    task_raw = [summary[summary["task"] == t]["raw_acc"].mean() for t in tasks]
    task_filt = [summary[summary["task"] == t]["filt_acc"].mean() for t in tasks]

    bars_r = ax_task.bar(
        x - width / 2,
        task_raw,
        width,
        label="Raw",
        color="#1976D2",
        alpha=0.85,
        edgecolor="white",
    )
    bars_f = ax_task.bar(
        x + width / 2,
        task_filt,
        width,
        label="Filtered",
        color="#E91E63",
        alpha=0.85,
        edgecolor="white",
    )
    _bar_label(ax_task, bars_r)
    _bar_label(ax_task, bars_f)

    # Trial dots (jittered)
    for xi, task in zip(x, tasks):
        t_rows = summary[summary["task"] == task]
        n = len(t_rows)
        jitter = np.linspace(-0.07, 0.07, n) if n > 1 else [0.0]
        for j, (_, row) in zip(jitter, t_rows.iterrows()):
            ax_task.scatter(
                xi - width / 2 + j,
                row["raw_acc"],
                color="#1976D2",
                s=35,
                zorder=5,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
            )
            ax_task.scatter(
                xi + width / 2 + j,
                row["filt_acc"],
                color="#E91E63",
                s=35,
                zorder=5,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
            )

    # x-axis labels include CWL level
    task_xlabels = []
    for task in tasks:
        lvl = int(summary[summary["task"] == task]["nback_level"].mode().iloc[0])
        n_trials = len(summary[summary["task"] == task])
        task_xlabels.append(f"{task}\n({STATE_LABELS[lvl]}, n={n_trials})")

    ax_task.set_xticks(x)
    ax_task.set_xticklabels(task_xlabels, fontsize=9)
    ax_task.set_ylim(0, 1.18)
    ax_task.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_task.set_ylabel("Accuracy")
    ax_task.set_title("Accuracy per Task")
    ax_task.legend(fontsize=8)
    ax_task.grid(axis="y", linestyle=":", alpha=0.4)
    ax_task.axhline(
        chance, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label="Chance"
    )

    # ── Panel B: per CWL level ───────────────────────────────────────────────
    levels = sorted(summary["nback_level"].unique())
    x2 = np.arange(len(levels))

    level_raw = [summary[summary["nback_level"] == l]["raw_acc"].mean() for l in levels]
    level_filt = [
        summary[summary["nback_level"] == l]["filt_acc"].mean() for l in levels
    ]
    level_counts = [int((summary["nback_level"] == l).sum()) for l in levels]

    bars_r2 = ax_level.bar(
        x2 - width / 2,
        level_raw,
        width,
        label="Raw",
        color="#1976D2",
        alpha=0.85,
        edgecolor="white",
    )
    bars_f2 = ax_level.bar(
        x2 + width / 2,
        level_filt,
        width,
        label="Filtered",
        color="#E91E63",
        alpha=0.85,
        edgecolor="white",
    )
    _bar_label(ax_level, bars_r2)
    _bar_label(ax_level, bars_f2)

    # Trial dots per level
    for xi, level in zip(x2, levels):
        l_rows = summary[summary["nback_level"] == level]
        n = len(l_rows)
        jitter = np.linspace(-0.07, 0.07, n) if n > 1 else [0.0]
        for j, (_, row) in zip(jitter, l_rows.iterrows()):
            ax_level.scatter(
                xi - width / 2 + j,
                row["raw_acc"],
                color="#1976D2",
                s=35,
                zorder=5,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
            )
            ax_level.scatter(
                xi + width / 2 + j,
                row["filt_acc"],
                color="#E91E63",
                s=35,
                zorder=5,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
            )

    ax_level.set_xticks(x2)
    ax_level.set_xticklabels(
        [f"{STATE_LABELS[l]}\n(n={level_counts[i]})" for i, l in enumerate(levels)],
        fontsize=9,
    )
    ax_level.set_ylim(0, 1.18)
    ax_level.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_level.set_ylabel("Accuracy")
    ax_level.set_title("Accuracy per CWL Level")
    ax_level.grid(axis="y", linestyle=":", alpha=0.4)
    ax_level.axhline(
        chance, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label="Chance"
    )
    ax_level.legend(fontsize=8, loc=1)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment-level accuracy summary (legacy, multiple subjects)
# ─────────────────────────────────────────────────────────────────────────────


def plot_inference_accuracy_summary(
    data: pd.DataFrame, ax_overall: plt.Axes, ax_per_class: plt.Axes
):
    gt = data["nback_level"].to_numpy()
    raw = data["raw_state"].to_numpy()
    filt = data["filtered_state"].to_numpy()

    acc_raw = (gt == raw).mean()
    acc_filt = (gt == filt).mean()

    bars = ax_overall.bar(
        ["Raw", "Filtered"],
        [acc_raw, acc_filt],
        color=["#1976D2", "#E91E63"],
        width=0.5,
        edgecolor="white",
    )
    ax_overall.set_ylim(0, 1.1)
    ax_overall.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_overall.set_ylabel("Accuracy")
    ax_overall.set_title("Overall Accuracy")
    ax_overall.grid(axis="y", linestyle=":", alpha=0.4)
    for bar, val in zip(bars, [acc_raw, acc_filt]):
        ax_overall.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    levels = [0, 1, 2]
    x = np.arange(len(levels))
    width = 0.35

    raw_per_class, filt_per_class, counts = [], [], []
    for lvl in levels:
        mask = gt == lvl
        n = mask.sum()
        counts.append(n)
        raw_per_class.append((raw[mask] == lvl).mean() if n > 0 else 0.0)
        filt_per_class.append((filt[mask] == lvl).mean() if n > 0 else 0.0)

    bars_raw = ax_per_class.bar(
        x - width / 2,
        raw_per_class,
        width,
        label="Raw",
        color="#1976D2",
        edgecolor="white",
    )
    bars_filt = ax_per_class.bar(
        x + width / 2,
        filt_per_class,
        width,
        label="Filtered",
        color="#E91E63",
        edgecolor="white",
    )
    ax_per_class.set_xticks(x)
    ax_per_class.set_xticklabels(
        [f"{STATE_LABELS[l]}\n(n={counts[i]})" for i, l in enumerate(levels)]
    )
    ax_per_class.set_ylim(0, 1.15)
    ax_per_class.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_per_class.set_ylabel("Accuracy")
    ax_per_class.set_title("Per-Class Accuracy")
    ax_per_class.legend(fontsize=8, loc=1)
    ax_per_class.grid(axis="y", linestyle=":", alpha=0.4)
    for bars_group in (bars_raw, bars_filt):
        _bar_label(ax_per_class, bars_group)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory CWL helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_spline() -> pd.DataFrame | None:
    """Load spline_trajectory.csv (x, z columns used for the track outline)."""
    if not _SPLINE_FILE.exists():
        print(f"  Warning: spline trajectory not found at {_SPLINE_FILE}")
        return None
    return pd.read_csv(_SPLINE_FILE)


def _load_trial_drone(trial_dir: Path) -> pd.DataFrame | None:
    """Load drone_data.csv from a trial folder, return mean position per timestamp."""
    drone_path = trial_dir / DRONE_FILE_NAME
    if not drone_path.exists():
        return None
    df = pd.read_csv(drone_path)
    if df.empty:
        return None
    # Mean position across all drone IDs per timestamp
    return (
        df.groupby("timestamp")[["position_x", "position_z"]]
        .mean()
        .reset_index()
        .sort_values("timestamp")
    )


def _load_trial_inference(trial_dir: Path) -> pd.DataFrame | None:
    """Load inference_data.csv from a trial folder."""
    inf_path = trial_dir / INFERENCE_FILE_NAME
    if not inf_path.exists():
        return None
    df = pd.read_csv(inf_path)
    return df.sort_values("timestamp") if not df.empty else None


def _join_cwl_to_drone(drone_df: pd.DataFrame, inf_df: pd.DataFrame) -> pd.DataFrame:
    """Attach filtered_state and nback_level to each drone row via merge_asof."""
    drone_sorted = drone_df.sort_values("timestamp")
    inf_sorted = inf_df[["timestamp", "filtered_state", "nback_level"]].sort_values(
        "timestamp"
    )
    return pd.merge_asof(
        drone_sorted, inf_sorted, on="timestamp", direction="backward"
    ).dropna(subset=["filtered_state"])


def _draw_spline_background(ax: plt.Axes, spline_df: pd.DataFrame):
    """Draw the track as a gray dashed background and set axis limits."""
    sx, sz = spline_df["x"].values, spline_df["z"].values
    ax.plot(sz, sx, color="lightgray", linewidth=2, linestyle="--", zorder=0)
    pad_x = (sx.max() - sx.min()) * 0.1
    pad_z = (sz.max() - sz.min()) * 0.1
    ax.set_ylim(sx.min() - pad_x, sx.max() + pad_x)
    # Invert horizontal axis for clockwise motion (matching visualize.py)
    ax.set_xlim(sz.max() + pad_z, sz.min() - pad_z)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Z (m)")
    ax.set_ylabel("X (m)")


def _compute_arc_param(spline_df: pd.DataFrame) -> np.ndarray:
    """Cumulative arc-length along the spline, normalised to [0, 1]."""
    sx, sz = spline_df["x"].values, spline_df["z"].values
    ds = np.sqrt(np.diff(sx) ** 2 + np.diff(sz) ** 2)
    arc = np.concatenate([[0.0], np.cumsum(ds)])
    return arc / arc[-1]


def _project_to_arc(
    pos_x: np.ndarray,
    pos_z: np.ndarray,
    spline_df: pd.DataFrame,
    arc_param: np.ndarray,
) -> np.ndarray:
    """Project (x, z) positions onto the nearest spline point's arc parameter."""
    sx, sz = spline_df["x"].values, spline_df["z"].values
    tree = cKDTree(np.column_stack([sz, sx]))  # query in (z, x) order
    _, idx = tree.query(np.column_stack([pos_z, pos_x]))
    return arc_param[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Per-CWL-level checkbox widget
# ─────────────────────────────────────────────────────────────────────────────


def _add_cwl_checkboxes(
    ax_traj: plt.Axes,
    per_level_artists: dict[int, list],
    mean_artists: list,
) -> None:
    """Add Low / Medium / High CheckButtons to the trajectory figure.

    All three are checked by default (matches the initial full view).
    When only a subset is checked, individual traces for those levels are
    shown and *mean_artists* (aggregate mean line) are hidden — the mean
    only makes sense when all levels are visible.
    The widget reference is pinned to the figure to prevent GC.
    """
    fig = ax_traj.figure
    labels = [STATE_LABELS[lvl] for lvl in (0, 1, 2)]
    # Taller axes to fit three rows
    cb_ax = fig.add_axes([0.01, 0.01, 0.09, 0.14])
    cb_ax.set_facecolor("#f8f8f8")
    cb = CheckButtons(cb_ax, labels, [True, True, True])
    for i, lbl in enumerate(cb.labels):
        lbl.set_fontsize(9)
        lbl.set_color(STATE_COLORS[i])

    def _on_toggle(_label: str) -> None:
        statuses = cb.get_status()
        all_checked = all(statuses)
        for i, level in enumerate((0, 1, 2)):
            vis = statuses[i]
            for artist in per_level_artists.get(level, []):
                artist.set_visible(vis)
        for artist in mean_artists:
            artist.set_visible(all_checked)
        fig.canvas.draw_idle()

    cb.on_clicked(_on_toggle)
    fig._cwl_checkboxes = cb  # keep alive


# ─────────────────────────────────────────────────────────────────────────────
# Subject + task trajectory plot
# ─────────────────────────────────────────────────────────────────────────────


def _plot_subject_task_trajectory(
    data_dir: Path,
    cwl_level: int,
    spline_df: pd.DataFrame,
    ax_traj: plt.Axes,
    ax_acc: plt.Axes,
):
    """Left: trajectory colored by CWL per trial.  Right: per-trial accuracy bars."""
    task = _find_task_for_cwl(data_dir, cwl_level)
    if task is None:
        cwl_label = STATE_LABELS.get(cwl_level, str(cwl_level))
        print(f"  No task found for CWL level {cwl_label} under {data_dir}")
        return
    task_dir = data_dir / task
    if not task_dir.exists():
        print(f"  Task folder not found: {task_dir}")
        return

    trial_dirs = sorted(
        d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")
    )
    if not trial_dirs:
        print(f"  No trial folders found under {task_dir}")
        return

    _draw_spline_background(ax_traj, spline_df)

    # Collect accuracy data for the right panel
    trial_names, raw_accs, filt_accs, gt_levels = [], [], [], []
    drawn_levels: set[int] = set()
    per_level_artists: dict[int, list] = {0: [], 1: [], 2: []}

    for trial_dir in trial_dirs:
        drone_df = _load_trial_drone(trial_dir)
        inf_df = _load_trial_inference(trial_dir)
        if drone_df is None or inf_df is None:
            continue

        merged = _join_cwl_to_drone(drone_df, inf_df)
        if merged.empty:
            continue

        # Scatter drone positions colored by filtered CWL
        for level, color in STATE_COLORS.items():
            mask = merged["filtered_state"] == level
            if not mask.any():
                continue
            sub = merged[mask]
            label = STATE_LABELS[level] if level not in drawn_levels else "_"
            drawn_levels.add(level)
            sc = ax_traj.scatter(
                sub["position_z"],
                sub["position_x"],
                c=color,
                s=6,
                alpha=0.6,
                label=label,
                zorder=2,
            )
            per_level_artists[level].append(sc)

        # Mark trial start
        first = merged.iloc[0]
        ax_traj.annotate(
            trial_dir.name,
            (first["position_z"], first["position_x"]),
            fontsize=6,
            color="#555",
            ha="center",
            va="bottom",
            textcoords="offset points",
            xytext=(0, 4),
        )

        # Accuracy for right panel
        gt_arr = inf_df["nback_level"].to_numpy()
        raw_arr = inf_df["raw_state"].to_numpy()
        filt_arr = inf_df["filtered_state"].to_numpy()
        trial_names.append(trial_dir.name)
        raw_accs.append(float((gt_arr == raw_arr).mean()))
        filt_accs.append(float((gt_arr == filt_arr).mean()))
        gt_levels.append(int(pd.Series(gt_arr).mode().iloc[0]))

    _add_cwl_checkboxes(ax_traj, per_level_artists, [])

    gt_level = gt_levels[0] if gt_levels else -1
    gt_label = STATE_LABELS.get(gt_level, "?")
    cwl_label = STATE_LABELS.get(cwl_level, str(cwl_level))
    ax_traj.set_title(f"Trajectory — CWL: {cwl_label} ({task}) (GT: {gt_label})")
    ax_traj.legend(loc="upper right", fontsize=7, markerscale=2)

    # ── Right panel: per-trial accuracy bars ─────────────────────────────────
    if not trial_names:
        ax_acc.text(0.5, 0.5, "No data", transform=ax_acc.transAxes, ha="center")
        return

    y = np.arange(len(trial_names))
    height = 0.35
    bars_r = ax_acc.barh(
        y - height / 2,
        raw_accs,
        height,
        label="Raw",
        color="#1976D2",
        alpha=0.85,
        edgecolor="white",
    )
    bars_f = ax_acc.barh(
        y + height / 2,
        filt_accs,
        height,
        label="Filtered",
        color="#E91E63",
        alpha=0.85,
        edgecolor="white",
    )
    _hbar_label(ax_acc, bars_r)
    _hbar_label(ax_acc, bars_f)

    ax_acc.set_yticks(y)
    ax_acc.set_yticklabels(trial_names, fontsize=9)
    ax_acc.set_xlim(0, 1.15)
    ax_acc.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_acc.set_xlabel("Accuracy")
    ax_acc.set_title(f"Per-Trial Accuracy — CWL: {cwl_label} ({task})")
    ax_acc.legend(fontsize=8, loc="lower right")
    ax_acc.grid(axis="x", linestyle=":", alpha=0.4)
    ax_acc.axvline(1 / 3, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_acc.invert_yaxis()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment + task aggregate trajectory plot
# ─────────────────────────────────────────────────────────────────────────────


def _plot_aggregate_task_trajectory(
    data_dir: Path,
    cwl_level: int,
    spline_df: pd.DataFrame,
    ax_traj: plt.Axes,
    ax_acc: plt.Axes,
):
    """Left: aggregate trajectory (mean ± std).  Right: per-subject accuracy."""
    cwl_label = STATE_LABELS.get(cwl_level, str(cwl_level))

    # Discover all subjects (task order may differ per subject)
    subject_dirs = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and _SUBJECT_RE.match(d.name)
    )
    if not subject_dirs:
        print(f"  No subject folders found under {data_dir}")
        return

    _draw_spline_background(ax_traj, spline_df)
    arc_param = _compute_arc_param(spline_df)

    # Per-subject: collect individual traces + accuracy
    subject_names, subject_filt_accs, subject_raw_accs = [], [], []
    all_arc_cwl: list[pd.DataFrame] = []  # for binning
    drawn_levels: set[int] = set()
    per_level_artists: dict[int, list] = {0: [], 1: [], 2: []}
    mean_artists: list = []

    for subj_dir in subject_dirs:
        # Resolve which task holds this CWL level for this subject
        task = _find_task_for_cwl(subj_dir, cwl_level)
        if task is None:
            print(f"  {subj_dir.name}: no task found for CWL={cwl_label}, skipping.")
            continue
        task_dir = subj_dir / task
        trial_dirs = sorted(
            d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")
        )
        if not trial_dirs:
            continue

        subj_merged_frames = []
        subj_gt, subj_raw, subj_filt = [], [], []

        for trial_dir in trial_dirs:
            drone_df = _load_trial_drone(trial_dir)
            inf_df = _load_trial_inference(trial_dir)
            if drone_df is None or inf_df is None:
                continue
            merged = _join_cwl_to_drone(drone_df, inf_df)
            if merged.empty:
                continue
            subj_merged_frames.append(merged)
            subj_gt.extend(inf_df["nback_level"].tolist())
            subj_raw.extend(inf_df["raw_state"].tolist())
            subj_filt.extend(inf_df["filtered_state"].tolist())

        if not subj_merged_frames:
            continue

        subj_all = pd.concat(subj_merged_frames, ignore_index=True)

        # Individual trace (faint)
        for level, color in STATE_COLORS.items():
            mask = subj_all["filtered_state"] == level
            if not mask.any():
                continue
            sub = subj_all[mask]
            label = STATE_LABELS[level] if level not in drawn_levels else "_"
            drawn_levels.add(level)
            sc = ax_traj.scatter(
                sub["position_z"],
                sub["position_x"],
                c=color,
                s=4,
                alpha=0.15,
                label=label,
                zorder=1,
            )
            per_level_artists[level].append(sc)

        # Project onto arc for binned aggregation
        arcs = _project_to_arc(
            subj_all["position_x"].values,
            subj_all["position_z"].values,
            spline_df,
            arc_param,
        )
        subj_all = subj_all.copy()
        subj_all["arc"] = arcs
        arc_cols = ["arc", "position_x", "position_z", "filtered_state"]
        all_arc_cwl.append(subj_all[arc_cols])

        # Accuracy
        subj_gt_arr = np.array(subj_gt)
        subj_raw_arr = np.array(subj_raw)
        subj_filt_arr = np.array(subj_filt)
        subject_names.append(f"{subj_dir.name} ({task})")
        subject_raw_accs.append(float((subj_gt_arr == subj_raw_arr).mean()))
        subject_filt_accs.append(float((subj_gt_arr == subj_filt_arr).mean()))

    # ── Mean trajectory (thick colored line) ─────────────────────────────────
    if all_arc_cwl:
        combined = pd.concat(all_arc_cwl, ignore_index=True)
        n_bins = 200
        combined["arc_bin"] = pd.cut(combined["arc"], bins=n_bins, labels=False)
        binned = (
            combined.groupby("arc_bin")
            .agg(
                x_mean=("position_x", "mean"),
                z_mean=("position_z", "mean"),
                x_std=("position_x", "std"),
                z_std=("position_z", "std"),
                cwl_mode=(
                    "filtered_state",
                    lambda s: int(s.mode().iloc[0]) if len(s) > 0 else 0,
                ),
            )
            .dropna()
        )

        # Draw mean line, colored per-bin
        for _, row in binned.iterrows():
            color = STATE_COLORS.get(int(row["cwl_mode"]), "#999")
            sc = ax_traj.scatter(
                row["z_mean"],
                row["x_mean"],
                c=color,
                s=30,
                zorder=3,
                edgecolors="white",
                linewidths=0.3,
            )
            mean_artists.append(sc)

    _add_cwl_checkboxes(ax_traj, per_level_artists, mean_artists)

    n_subjects = len(subject_names)
    ax_traj.set_title(
        f"Aggregate Trajectory — CWL: {cwl_label} (n={n_subjects} subjects)"
    )
    ax_traj.legend(loc="upper right", fontsize=7, markerscale=2)

    # ── Right panel: per-subject accuracy ────────────────────────────────────
    if not subject_names:
        ax_acc.text(0.5, 0.5, "No data", transform=ax_acc.transAxes, ha="center")
        return

    y = np.arange(len(subject_names))
    height = 0.35
    bars_r = ax_acc.barh(
        y - height / 2,
        subject_raw_accs,
        height,
        label="Raw",
        color="#1976D2",
        alpha=0.85,
        edgecolor="white",
    )
    bars_f = ax_acc.barh(
        y + height / 2,
        subject_filt_accs,
        height,
        label="Filtered",
        color="#E91E63",
        alpha=0.85,
        edgecolor="white",
    )
    _hbar_label(ax_acc, bars_r)
    _hbar_label(ax_acc, bars_f)

    ax_acc.set_yticks(y)
    ax_acc.set_yticklabels(subject_names, fontsize=9)
    ax_acc.set_xlim(0, 1.15)
    ax_acc.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_acc.set_xlabel("Accuracy")
    ax_acc.set_title(f"Per-Subject Accuracy — CWL: {cwl_label}")
    ax_acc.legend(fontsize=8, loc="upper right")
    ax_acc.grid(axis="x", linestyle=":", alpha=0.4)
    ax_acc.axvline(1 / 3, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_acc.invert_yaxis()

    # Summary stats text
    mean_filt = np.mean(subject_filt_accs)
    std_filt = np.std(subject_filt_accs)
    mean_raw = np.mean(subject_raw_accs)
    std_raw = np.std(subject_raw_accs)
    ax_acc.text(
        0.98,
        0.02,
        f"Filtered: {mean_filt:.1%} ± {std_filt:.1%}\n"
        f"Raw: {mean_raw:.1%} ± {std_raw:.1%}",
        transform=ax_acc.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Spline accuracy ribbon plot
# ─────────────────────────────────────────────────────────────────────────────


def _collect_merged_frames(data_dir: Path, cwl_level: int) -> list[pd.DataFrame]:
    """Return all drone+inference merged DataFrames for *cwl_level*.

    Works for both subject dirs (4-char code) and experiment dirs (containing
    subject sub-folders).  Task assignment is resolved per subject so that
    randomised task order is handled transparently.
    """
    mode = _detect_mode(data_dir)
    subject_dirs = (
        [data_dir]
        if mode == "subject"
        else sorted(
            d for d in data_dir.iterdir() if d.is_dir() and _SUBJECT_RE.match(d.name)
        )
    )
    frames: list[pd.DataFrame] = []
    for subj_dir in subject_dirs:
        task = _find_task_for_cwl(subj_dir, cwl_level)
        if task is None:
            continue
        task_dir = subj_dir / task
        trial_dirs = sorted(
            d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")
        )
        for trial_dir in trial_dirs:
            drone_df = _load_trial_drone(trial_dir)
            inf_df = _load_trial_inference(trial_dir)
            if drone_df is None or inf_df is None:
                continue
            merged = _join_cwl_to_drone(drone_df, inf_df)
            if not merged.empty:
                frames.append(merged)
    return frames


def _plot_spline_accuracy_ribbon(
    ax: plt.Axes,
    spline_df: pd.DataFrame,
    all_merged_frames: list[pd.DataFrame],
    cwl_level: int,
    n_bins: int = 120,
) -> None:
    """Draw a per-class prediction ribbon on the spline.

    For each arc bin, three colored bands are drawn perpendicular to the
    track and stacked outward from the spline center (Low → Medium → High).
    The width of each band is proportional to the prediction count for that
    class.  Total stacked width at the busiest bin fills *max_hw*, so the
    dominant class at each segment is immediately visible.

    A thin gray centerline replaces the previous class-colored line.
    """
    if not all_merged_frames:
        return

    arc_param = _compute_arc_param(spline_df)
    sx = spline_df["x"].values
    sz = spline_df["z"].values

    # Spline tangent + normal in plot space (z on x-axis, x on y-axis)
    tgz = np.gradient(sz)
    tgx = np.gradient(sx)
    mag = np.sqrt(tgz**2 + tgx**2) + 1e-9
    tgz /= mag
    tgx /= mag
    nz = tgx  # 90° CW normal → points outside the track for a CW loop
    nx = -tgz

    # Project inference rows onto arc and bin
    combined = pd.concat(all_merged_frames, ignore_index=True)
    arcs = _project_to_arc(
        combined["position_x"].values,
        combined["position_z"].values,
        spline_df,
        arc_param,
    )
    combined = combined.copy()
    combined["arc"] = arcs
    combined["arc_bin"] = pd.cut(combined["arc"], bins=n_bins, labels=False)

    # Per-bin count of each predicted class
    counts_df = (
        combined.groupby(["arc_bin", "filtered_state"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1, 2], fill_value=0)
    )
    if counts_df.empty:
        return

    # Normalize: max total across all bins → max_hw
    totals = counts_df.sum(axis=1)
    max_total = totals.max() or 1
    track_extent = min(sz.max() - sz.min(), sx.max() - sx.min())
    max_hw = track_extent * 0.06  # total stacked width at busiest bin

    # Map each spline index to its arc bin
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    spline_bin = np.clip(
        np.searchsorted(bin_edges, arc_param, side="right") - 1, 0, n_bins - 1
    )

    def _stacked_quad(i: int, j: int, inner: float, outer: float):
        """Return the four corners of a quad offset by [inner, outer] along the normal."""
        return [
            (sz[i] + nz[i] * inner, sx[i] + nx[i] * inner),
            (sz[j] + nz[j] * inner, sx[j] + nx[j] * inner),
            (sz[j] + nz[j] * outer, sx[j] + nx[j] * outer),
            (sz[i] + nz[i] * outer, sx[i] + nx[i] * outer),
        ]

    # ── Per-class stacked ribbons ─────────────────────────────────────────────
    # Draw each spline segment; for each segment look up bin counts, compute
    # cumulative offsets [0 → w0 → w0+w1 → w0+w1+w2] on the normal direction.
    for i in range(len(sz) - 1):
        b = int(spline_bin[i])
        if b not in counts_df.index:
            continue
        row = counts_df.loc[b]
        offset = 0.0
        for level in (0, 1, 2):
            w = max_hw * float(row[level]) / max_total
            if w < 1e-6:
                offset += w
                continue
            verts = _stacked_quad(i, i + 1, offset, offset + w)
            ax.add_patch(
                MplPolygon(
                    verts,
                    closed=True,
                    facecolor=STATE_COLORS[level],
                    alpha=0.55,
                    linewidth=0,
                    zorder=2,
                )
            )
            offset += w

    # ── Neutral gray centerline ───────────────────────────────────────────────
    segments = [[(sz[i], sx[i]), (sz[i + 1], sx[i + 1])] for i in range(len(sz) - 1)]
    lc = LineCollection(segments, colors="#555555", linewidths=1.5, zorder=3)
    ax.add_collection(lc)

    # ── Legend ────────────────────────────────────────────────────────────────
    cwl_label = STATE_LABELS.get(cwl_level, str(cwl_level))
    for level, color in STATE_COLORS.items():
        ax.scatter(
            [],
            [],
            c=color,
            s=80,
            marker="s",
            alpha=0.55,
            label=f"{STATE_LABELS[level]} predicted",
        )

    ax.set_title(
        f"Prediction Ribbon — CWL: {cwl_label}\n"
        "Stacked bands per arc segment · width ∝ prediction count"
    )
    ax.legend(loc="upper right", fontsize=8, markerscale=1.2)


# ─────────────────────────────────────────────────────────────────────────────
# Entry points per mode
# ─────────────────────────────────────────────────────────────────────────────


def _save_or_show(figs: list[tuple[plt.Figure, Path]], show: bool):
    for fig, path in figs:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    if show:
        plt.show()
    else:
        plt.close("all")


def _make_time_series_figure(data: pd.DataFrame, title: str) -> plt.Figure:
    fig, (ax_states, ax_rolling) = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plot_inference_time_series(data, ax_states, ax_rolling)
    fig.tight_layout()
    return fig


def run_inference(
    show: bool,
    output_dir: Path,
    data_dir: Path,
    cwl: int | None = None,
):
    mode = _detect_mode(data_dir)
    print(f"Loading inference data from: {data_dir}  [{mode} mode]")

    output_dir.mkdir(parents=True, exist_ok=True)
    subject = data_dir.name
    figs: list[tuple[plt.Figure, Path]] = []

    # ── CWL trajectory mode ──────────────────────────────────────────────────
    if cwl is not None:
        cwl_label = STATE_LABELS.get(cwl, str(cwl)).lower()
        spline_df = _load_spline()
        if spline_df is None:
            print("  Cannot produce trajectory plots without spline_trajectory.csv")
            return

        if mode == "subject":
            fig, (ax_traj, ax_acc) = plt.subplots(
                1,
                2,
                figsize=(16, 7),
                gridspec_kw={"width_ratios": [2, 1], "wspace": 0.25},
            )
            fig.suptitle(
                f"CWL Trajectory — {subject} — {STATE_LABELS.get(cwl, cwl)}",
                fontsize=13,
                fontweight="bold",
            )
            _plot_subject_task_trajectory(data_dir, cwl, spline_df, ax_traj, ax_acc)
            fig.tight_layout()
            figs.append((fig, output_dir / f"trajectory_cwl_{cwl_label}.png"))

        elif mode == "experiment":
            fig, (ax_traj, ax_acc) = plt.subplots(
                1,
                2,
                figsize=(16, 7),
                gridspec_kw={"width_ratios": [2, 1], "wspace": 0.25},
            )
            fig.suptitle(
                f"Aggregate CWL Trajectory — {STATE_LABELS.get(cwl, cwl)}",
                fontsize=13,
                fontweight="bold",
            )
            _plot_aggregate_task_trajectory(
                data_dir,
                cwl,
                spline_df,
                ax_traj,
                ax_acc,
            )
            fig.tight_layout()
            out_name = f"trajectory_cwl_aggregate_{cwl_label}.png"
            figs.append((fig, output_dir / out_name))

        # ── Accuracy ribbon plot (both modes) ────────────────────────────────
        merged_frames = _collect_merged_frames(data_dir, cwl)
        if merged_frames:
            fig_r, ax_r = plt.subplots(figsize=(10, 9))
            fig_r.suptitle(
                f"Accuracy Ribbon — {STATE_LABELS.get(cwl, cwl)}"
                + (f" — {subject}" if mode == "subject" else ""),
                fontsize=13,
                fontweight="bold",
            )
            _draw_spline_background(ax_r, spline_df)
            _plot_spline_accuracy_ribbon(ax_r, spline_df, merged_frames, cwl)
            fig_r.tight_layout()
            ribbon_name = f"trajectory_cwl_ribbon_{cwl_label}.png"
            figs.append((fig_r, output_dir / ribbon_name))

        else:
            print(f"  --cwl is not supported in {mode} mode.")
            return

        _save_or_show(figs, show)
        return

    # ── Standard mode (no --task) ────────────────────────────────────────────
    data = load_inference_data(data_dir)
    n_sources = data["_source"].nunique()
    print(f"  Loaded {len(data)} rows from {n_sources} session(s).")

    if mode == "trial":
        fig = _make_time_series_figure(data, f"Workload Inference — {subject}")
        figs.append((fig, output_dir / "inference_time_series.png"))

    elif mode == "subject":
        task_data = _task_trials_only(data)
        if task_data.empty:
            print("  No task trial data found — skipping time series.")
        else:
            fig1 = _make_time_series_figure(
                task_data, f"Workload Inference — {subject} — Task Trials"
            )
            figs.append((fig1, output_dir / "inference_time_series.png"))

            fig2, (ax_task, ax_level) = plt.subplots(1, 2, figsize=(12, 5))
            fig2.suptitle(
                f"Workload Inference — {subject} — Accuracy Summary",
                fontsize=13,
                fontweight="bold",
            )
            plot_subject_accuracy_summary(task_data, ax_task, ax_level)
            fig2.tight_layout()
            figs.append((fig2, output_dir / "inference_accuracy_summary.png"))

    else:
        fig1 = _make_time_series_figure(
            data, "Real-Time Workload Inference — Time Series"
        )
        figs.append((fig1, output_dir / "inference_time_series.png"))

        fig2, (ax_overall, ax_per_class) = plt.subplots(1, 2, figsize=(10, 5))
        fig2.suptitle(
            "Real-Time Workload Inference — Accuracy Summary",
            fontsize=13,
            fontweight="bold",
        )
        plot_inference_accuracy_summary(data, ax_overall, ax_per_class)
        fig2.tight_layout()
        figs.append((fig2, output_dir / "inference_accuracy_summary.png"))

    _save_or_show(figs, show)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Racing experiment
# ─────────────────────────────────────────────────────────────────────────────

GATE_LAYOUT_FILE = "gate_layout.csv"
GATE_STATUS_FILE = "gate_status.csv"
COMMAND_DATA_FILE = "command_data.csv"
SWARM_SIZE = 9

# Penalty weights for the composite trial-performance score (s-equivalent).
# Tune these to weight crash/miss severity vs lap time.
RACING_DEAD_PENALTY_S = 30.0
RACING_MISS_PENALTY_S = 5.0

# Adaptation-group visualisation
GROUP_COLORS = {
    "adaptive": "#1976D2",
    "non_adaptive": "#FB8C00",
    "unknown": "#9E9E9E",
}
GROUP_LABELS = {
    "adaptive": "Adaptive",
    "non_adaptive": "Non-Adaptive (Control)",
    "unknown": "Unknown",
}

COMMAND_INPUTS = {
    "pitch_rate": ("#1976D2", "Pitch"),
    "yaw_rate": ("#E91E63", "Yaw"),
    "roll_rate": ("#FF9800", "Roll"),
    "altitude_rate": ("#4CAF50", "Altitude"),
}

ADAPTATION_PARAMS = {
    "max_speed": ("Max Speed", 3.0, 15.0),
    "max_yaw_rate": ("Max Yaw Rate", 0.6, 1.5),
    "max_pitch": ("Max Pitch", 0.15, 0.4),
    "max_roll": ("Max Roll", 0.15, 0.4),
    "max_ascent_rate": ("Max Ascent", 1.5, 3.0),
    "max_descent_rate": ("Max Descent", 1.5, 2.5),
    "max_altitude_rate": ("Max Altitude Rate", 1.5, 3.0),
    "max_alpha": ("Max Alpha", 6.0, 15.0),
}

DIFFICULTY_COL = "is_hard"

# Flight profile limits (mirrors plot_command_limits.py constants)
_FLIGHT_PROFILE_LIMITS: dict[str, float] = {
    "max_pitch": 0.45,
    "max_roll": 0.45,
    "max_yaw_rate": 1.5,
    "max_speed": 15.0,
    "max_altitude_rate": 5.0,
    "max_alpha": 12.0,
}
_FLIGHT_PROFILE_MIN_LIMITS: dict[str, float] = {
    "max_pitch": 0.2,
    "max_roll": 0.2,
    "max_yaw_rate": 0.6,
    "max_speed": 3.0,
    "max_altitude_rate": 1.5,
    "max_alpha": 5.0,
}
_FLIGHT_PARAM_LABELS: dict[str, tuple[str, str]] = {
    "max_pitch": ("Max Pitch", "rad"),
    "max_roll": ("Max Roll", "rad"),
    "max_yaw_rate": ("Max Yaw Rate", "rad/s"),
    "max_speed": ("Max Speed", "m/s"),
    "max_altitude_rate": ("Max Altitude Rate", "m/s"),
    "max_alpha": ("Max Alpha", "°"),
}


def _load_csv(trial_dir: Path, filename: str) -> pd.DataFrame | None:
    path = trial_dir / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def _compute_centroid(drones: pd.DataFrame) -> pd.DataFrame | None:
    valid = drones[drones["timestamp"] > 0]
    if "alive" in valid.columns:
        alive = valid[valid["alive"] > 0]
        if not alive.empty:
            valid = alive
    if valid.empty:
        return None
    centroid = (
        valid.groupby("timestamp")
        .agg(
            x=("position_x", "mean"),
            y=("position_y", "mean"),
            z=("position_z", "mean"),
            n_alive=("id", "nunique"),
        )
        .reset_index()
        .sort_values("timestamp")
    )
    t0 = centroid["timestamp"].iloc[0]
    centroid["elapsed_s"] = (centroid["timestamp"] - t0) / 1000.0
    return centroid


def _gate_passage_times(gate_status: pd.DataFrame) -> pd.DataFrame:
    passed = gate_status[gate_status["first_pass_timestamp"] > 0].copy()
    return passed.sort_values("first_pass_timestamp")


def _find_t0(inference, centroid, commands, gate_status) -> int:
    candidates = []
    for df in [inference, centroid, commands]:
        if df is not None and "timestamp" in df.columns:
            valid_ts = df[df["timestamp"] > 0]["timestamp"]
            if not valid_ts.empty:
                candidates.append(int(valid_ts.min()))
    if gate_status is not None:
        valid_ts = gate_status[gate_status["first_pass_timestamp"] > 0][
            "first_pass_timestamp"
        ]
        if not valid_ts.empty:
            candidates.append(int(valid_ts.min()))
    return min(candidates) if candidates else 0


def _shade_difficulty_time(ax, gates, gate_status, t0_ms):
    if DIFFICULTY_COL not in gates.columns or gate_status is None:
        return
    passed = _gate_passage_times(gate_status)
    if len(passed) < 2:
        return
    gate_info = gates.set_index("id")
    pts = []
    for _, row in passed.iterrows():
        gid = int(row["id"])
        if gid in gate_info.index:
            hard = bool(gate_info.loc[gid, DIFFICULTY_COL])
            t_s = (row["first_pass_timestamp"] - t0_ms) / 1000.0
            pts.append((t_s, hard))
    for i in range(len(pts) - 1):
        t0_s, _ = pts[i]
        t1_s, hard = pts[i + 1]
        color = "#F44336" if hard else "#4CAF50"
        ax.axvspan(t0_s, t1_s, alpha=0.08, color=color, linewidth=0)


def _shade_difficulty_z(ax, gates, alpha: float = 0.06):
    if DIFFICULTY_COL not in gates.columns:
        return
    sorted_gates = gates.sort_values("center_z")
    for i in range(len(sorted_gates) - 1):
        g_next = sorted_gates.iloc[i + 1]
        hard = bool(g_next[DIFFICULTY_COL])
        color = "#F44336" if hard else "#4CAF50"
        ax.axvspan(
            sorted_gates.iloc[i]["center_z"],
            g_next["center_z"],
            alpha=alpha,
            color=color,
            linewidth=0,
        )


def _draw_gate_lines(ax, gate_status, t0_ms):
    passed = _gate_passage_times(gate_status)
    for _, row in passed.iterrows():
        t_s = (row["first_pass_timestamp"] - t0_ms) / 1000.0
        ax.axvline(t_s, color="#aaa", linewidth=0.5, linestyle=":", alpha=0.5)


def _draw_gate_lines_z(ax, gates):
    for _, g in gates.sort_values("center_z").iterrows():
        ax.axvline(g["center_z"], color="#aaa", linewidth=0.5, linestyle=":", alpha=0.5)


def _alive_at_timestamp(drones: pd.DataFrame, ts: int) -> int:
    """Return count of unique alive drone IDs at or just before ts."""
    if drones is None or drones.empty:
        return SWARM_SIZE
    before = drones[drones["timestamp"] <= ts]
    if before.empty:
        return SWARM_SIZE
    latest_ts = int(before["timestamp"].max())
    frame = before[before["timestamp"] == latest_ts]
    return int(frame[frame["alive"] > 0]["id"].nunique())


def _compute_gate_breakdown(gates, gate_status, drones) -> dict:
    """Returns mapping gate_id -> {inside, outside, dead, reached, first_pass_ts}."""
    breakdown = {}
    if gate_status is None:
        for _, g in gates.iterrows():
            breakdown[int(g["id"])] = dict(
                inside=0, outside=0, dead=SWARM_SIZE, reached=False, first_pass_ts=0
            )
        return breakdown
    for _, row in gate_status.iterrows():
        gid = int(row["id"])
        ts = int(row.get("first_pass_timestamp", 0))
        pc = int(row.get("pass_count", 0))
        reached = ts > 0
        alive_ids = _alive_at_timestamp(drones, ts) if (reached and drones is not None) else SWARM_SIZE
        dead = max(0, SWARM_SIZE - alive_ids)
        outside = max(0, alive_ids - pc)
        breakdown[gid] = dict(
            inside=pc, outside=outside, dead=dead, reached=reached, first_pass_ts=ts
        )
    return breakdown


def _time_to_z(timestamps_ms, centroid: pd.DataFrame) -> np.ndarray:
    return np.interp(
        np.asarray(timestamps_ms, dtype=float),
        centroid["timestamp"].values.astype(float),
        centroid["z"].values.astype(float),
    )


def _find_dead_drones(drones: pd.DataFrame) -> pd.DataFrame:
    if drones is None or drones.empty or "alive" not in drones.columns:
        return pd.DataFrame(columns=["id", "position_x", "position_z", "timestamp"])
    deaths = []
    for drone_id, grp in drones.groupby("id"):
        grp = grp.sort_values("timestamp")
        alive = grp["alive"].values
        if len(alive) == 0 or alive.min() != 0 or alive.max() == 0:
            continue
        # First index where alive==0 after being alive
        dead_idx = np.argmax(alive == 0)
        if dead_idx == 0:
            continue  # was never alive
        last_alive = grp.iloc[dead_idx - 1]
        deaths.append(
            {
                "id": int(drone_id),
                "position_x": float(last_alive["position_x"]),
                "position_z": float(last_alive["position_z"]),
                "timestamp": int(last_alive["timestamp"]),
            }
        )
    return pd.DataFrame(deaths)


def _no_data_placeholder(ax, title: str):
    ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=10)
    ax.set_title(title)


def _plot_racing_combined(
    ax,
    gates,
    centroid,
    traj_merged,
    gate_breakdown: dict,
    dead_drones=None,
    traj_type: str = "inference",
    gate_status=None,
):
    from matplotlib.collections import LineCollection as _LC
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap

    _shade_difficulty_z(ax, gates, alpha=0.10)

    # Gate bars — colored by pass status
    for _, g in gates.iterrows():
        gid = int(g["id"])
        half_w = g["width"] / 2
        bd = gate_breakdown.get(gid, {})
        reached = bd.get("reached", False)
        outside = bd.get("outside", 0)
        if not reached:
            bar_color = "#BDBDBD"  # gray = unreached
        elif outside > 0:
            bar_color = "#FFC107"  # yellow = some outside
        else:
            bar_color = "#4CAF50"  # green = all inside
        ax.plot(
            [g["center_z"], g["center_z"]],
            [g["center_x"] - half_w, g["center_x"] + half_w],
            color=bar_color,
            linewidth=2.5,
            alpha=0.85,
            solid_capstyle="round",
        )

    # Split times
    split_map = {}
    if gate_status is not None:
        passed = _gate_passage_times(gate_status)
        ts_list = passed["first_pass_timestamp"].values
        for i, (_, row) in enumerate(passed.iterrows()):
            if i > 0:
                split_map[int(row["id"])] = (ts_list[i] - ts_list[i - 1]) / 1000.0

    # Gate labels
    for _, g in gates.iterrows():
        gid = int(g["id"])
        half_w = g["width"] / 2
        label_parts = [f"G{gid}"]
        if gid in split_map:
            label_parts.append(f"{split_map[gid]:.1f}s")

        ax.text(
            g["center_z"],
            g["center_x"] + half_w + 4,
            "\n".join(label_parts),
            fontsize=6,
            ha="center",
            va="bottom",
            color="#444",
            fontweight="bold" if gid in split_map else "normal",
        )

        bd = gate_breakdown.get(gid, {})
        outside = bd.get("outside", 0)
        if outside > 0 and bd.get("reached", False):
            ax.text(
                g["center_z"],
                g["center_x"] - half_w - 2,
                f"⚠{outside}",
                fontsize=7,
                ha="center",
                va="top",
                color="#E65100",
                fontweight="bold",
            )

    # Completion time annotation on end marker
    completion_str = None
    if gate_status is not None:
        passed_gates = _gate_passage_times(gate_status)
        if len(passed_gates) >= 2:
            t_start = passed_gates["first_pass_timestamp"].min()
            t_end = passed_gates["first_pass_timestamp"].max()
            elapsed_s = (t_end - t_start) / 1000.0
            mm = int(elapsed_s // 60)
            ss = elapsed_s % 60
            completion_str = f"{mm:02d}:{ss:04.1f}"

    # Trajectory
    if traj_merged is not None and not traj_merged.empty:
        z_vals = traj_merged["z"].values
        x_vals = traj_merged["x"].values

        if traj_type == "adaptive" and "cwl_current_step" in traj_merged.columns:
            total = traj_merged["cwl_total_steps"].replace(0, np.nan).fillna(24)
            step_norm = (traj_merged["cwl_current_step"] / total).clip(0, 1).values
            cmap = LinearSegmentedColormap.from_list(
                "adaptive", ["#F44336", "#FFC107", "#4CAF50"]
            )
            seg_colors = [cmap(float((step_norm[i] + step_norm[i + 1]) / 2)) for i in range(len(step_norm) - 1)]
        else:
            states = traj_merged["filtered_state"].values.astype(int)
            seg_colors = [STATE_COLORS.get(s, "#999") for s in states[:-1]]

        points = np.column_stack([z_vals, x_vals]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = _LC(segments, colors=seg_colors, linewidths=3, alpha=0.85, zorder=2)
        ax.add_collection(lc)

        ax.scatter(z_vals[0], x_vals[0], marker="o", s=80, color="#333", zorder=5, edgecolors="white", linewidths=1.5)
        ax.scatter(z_vals[-1], x_vals[-1], marker="s", s=80, color="#333", zorder=5, edgecolors="white", linewidths=1.5)

        if completion_str:
            ax.annotate(
                f"Finish\n{completion_str}",
                xy=(z_vals[-1], x_vals[-1]),
                xytext=(6, -14),
                textcoords="offset points",
                fontsize=7,
                color="#333",
                fontweight="bold",
            )
    elif centroid is not None:
        ax.plot(centroid["z"], centroid["x"], color="#555", linewidth=2, alpha=0.7, zorder=2)

    # Dead drone markers
    has_dead = dead_drones is not None and not dead_drones.empty
    if has_dead:
        for _, row in dead_drones.iterrows():
            ax.scatter(row["position_z"], row["position_x"], marker="x", color="#222", s=140, linewidths=2.8, zorder=6)
            ax.annotate(
                f"D{int(row['id'])}",
                xy=(row["position_z"], row["position_x"]),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=7,
                color="#222",
                fontweight="bold",
                zorder=7,
            )

    # Legend
    if traj_type == "adaptive":
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize, LinearSegmentedColormap as _LSC
        cmap = _LSC.from_list("adaptive", ["#F44336", "#FFC107", "#4CAF50"])
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, location="right", fraction=0.02, pad=0.02,
                     label="Adaptation (Soft → Racing)")
        handles = []
    else:
        handles = [
            Line2D([0], [0], color=STATE_COLORS[i], linewidth=3, label=STATE_LABELS[i])
            for i in range(3)
        ]
    handles += [
        Line2D([0], [0], color="#4CAF50", linewidth=2.5, alpha=0.85, label="All inside"),
        Line2D([0], [0], color="#FFC107", linewidth=2.5, alpha=0.85, label="Some outside"),
        Line2D([0], [0], color="#BDBDBD", linewidth=2.5, alpha=0.85, label="Unreached gate"),
        Line2D([0], [0], marker="o", color="#333", linewidth=0, markersize=7, markeredgecolor="white", label="Start"),
        Line2D([0], [0], marker="s", color="#333", linewidth=0, markersize=7, markeredgecolor="white", label="End"),
    ]
    if has_dead:
        handles.append(
            Line2D([0], [0], marker="x", color="#222", linewidth=0, markersize=9, markeredgewidth=2.5, label="Dead drone")
        )
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)

    ax.set_ylabel("X — Lateral (m)")
    title_type = "Adaptation Step" if traj_type == "adaptive" else "CWL"
    ax.set_title(f"Course Overview — Trajectory colored by {title_type}, split times at gates")
    ax.grid(alpha=0.2)


def _plot_racing_cwl_probability(ax, inference, centroid, gates):
    if (
        inference is None
        or len(inference) <= 1
        or centroid is None
        or centroid.empty
    ):
        _no_data_placeholder(ax, "CWL Probabilities")
        return

    inf = inference[inference["timestamp"] > 0].copy()
    required = {"prob_low", "prob_medium", "prob_high"}
    if inf.empty or not required.issubset(inf.columns):
        _no_data_placeholder(ax, "CWL Probabilities")
        return

    inf = inf.sort_values("timestamp")
    z_raw = _time_to_z(inf["timestamp"].values, centroid)

    # Resample to uniform Z grid to avoid temporal-density distortion
    z_grid = np.linspace(z_raw.min(), z_raw.max(), 500)
    prob_low = np.interp(z_grid, z_raw, inf["prob_low"].values)
    prob_med = np.interp(z_grid, z_raw, inf["prob_medium"].values)
    prob_high = np.interp(z_grid, z_raw, inf["prob_high"].values)

    _shade_difficulty_z(ax, gates)
    ax.stackplot(
        z_grid,
        prob_low,
        prob_med,
        prob_high,
        colors=[STATE_COLORS[0], STATE_COLORS[1], STATE_COLORS[2]],
        alpha=0.85,
        labels=[STATE_LABELS[0], STATE_LABELS[1], STATE_LABELS[2]],
    )
    _draw_gate_lines_z(ax, gates)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("CWL Probability")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(axis="y", linestyle=":", alpha=0.3)


def _plot_racing_adaptation_z(ax, commands, centroid, gates):
    if commands is None or centroid is None or centroid.empty:
        _no_data_placeholder(ax, "Adaptation Profile")
        return

    cmd = commands[commands["timestamp"] > 0].copy()
    if (
        cmd.empty
        or "cwl_current_step" not in cmd.columns
        or "cwl_total_steps" not in cmd.columns
    ):
        _no_data_placeholder(ax, "Adaptation Profile")
        return

    cmd = cmd.sort_values("timestamp")
    total = cmd["cwl_total_steps"].replace(0, np.nan)
    step_norm_raw = (cmd["cwl_current_step"] / total).fillna(0).clip(0, 1).values
    z_raw = _time_to_z(cmd["timestamp"].values, centroid)

    # Resample to uniform Z grid
    z_grid = np.linspace(z_raw.min(), z_raw.max(), 500)
    step_norm = np.interp(z_grid, z_raw, step_norm_raw)

    _shade_difficulty_z(ax, gates)
    ax.fill_between(z_grid, 0, step_norm, alpha=0.25, color="#5C6BC0", zorder=2)
    ax.plot(z_grid, step_norm, color="#3949AB", linewidth=1.8, zorder=3)

    _draw_gate_lines_z(ax, gates)
    ax.set_ylim(-0.05, 1.10)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["Soft", "Racing"])
    ax.set_ylabel("Profile")
    ax.grid(axis="y", linestyle=":", alpha=0.3)


def _plot_racing_drone_misses(ax, gates, gate_breakdown: dict):
    if gates is None or gates.empty:
        _no_data_placeholder(ax, "Drones Through Gate")
        return

    sorted_gates = gates.sort_values("center_z").reset_index(drop=True)
    z_centers = sorted_gates["center_z"].values

    mean_span = float(np.mean(np.diff(z_centers))) if len(z_centers) >= 2 else 1.0
    bar_width = 0.6 * mean_span if mean_span > 0 else 1.0

    _shade_difficulty_z(ax, gates)

    legend_labels: set[str] = set()

    for _, g in sorted_gates.iterrows():
        gid = int(g["id"])
        z = float(g["center_z"])
        bd = gate_breakdown.get(gid, {})
        reached = bd.get("reached", False)

        if not reached:
            ax.bar(
                z, SWARM_SIZE, width=bar_width * 0.25,
                color="#9E9E9E", alpha=0.25, edgecolor="none", zorder=2,
                label="Unreached" if "Unreached" not in legend_labels else None,
            )
            legend_labels.add("Unreached")
            continue

        inside = bd.get("inside", 0)
        outside = bd.get("outside", 0)
        dead = bd.get("dead", 0)
        bottom = 0

        if inside > 0:
            ax.bar(z, inside, width=bar_width, bottom=bottom, color="#4CAF50",
                   edgecolor="white", linewidth=0.4, zorder=3,
                   label="Inside" if "Inside" not in legend_labels else None)
            legend_labels.add("Inside")
            bottom += inside

        if outside > 0:
            ax.bar(z, outside, width=bar_width, bottom=bottom, color="#FFC107",
                   edgecolor="white", linewidth=0.4, zorder=3,
                   label="Outside" if "Outside" not in legend_labels else None)
            legend_labels.add("Outside")
            bottom += outside

        if dead > 0:
            ax.bar(z, dead, width=bar_width, bottom=bottom, color="#212121",
                   edgecolor="white", linewidth=0.4, zorder=3,
                   label="Dead" if "Dead" not in legend_labels else None)
            legend_labels.add("Dead")

    _draw_gate_lines_z(ax, gates)
    ax.set_ylim(0, SWARM_SIZE + 0.5)
    ax.set_yticks([0, 3, 6, 9])
    ax.set_ylabel(f"Drones (/{SWARM_SIZE})")
    ax.set_xlabel("Course Progress Z (m)")
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    ax.grid(axis="y", linestyle=":", alpha=0.3)


def _plot_racing_cwl(ax, inference, gates, gate_status, t0_ms):
    if inference is None or len(inference) <= 1:
        _no_data_placeholder(ax, "CWL Inference")
        return

    inf = inference[inference["timestamp"] > 0].copy()
    if inf.empty:
        _no_data_placeholder(ax, "CWL Inference")
        return

    t = (inf["timestamp"] - t0_ms) / 1000.0

    if gates is not None and gate_status is not None:
        _shade_difficulty_time(ax, gates, gate_status, t0_ms)

    ax.step(
        t,
        inf["raw_state"],
        where="post",
        color="#1976D2",
        linewidth=1,
        alpha=0.5,
        label="Raw",
    )
    ax.step(
        t,
        inf["filtered_state"],
        where="post",
        color="#E91E63",
        linewidth=1.5,
        label="Filtered",
    )

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([STATE_LABELS[i] for i in range(3)])
    ax.set_ylabel("CWL Level")
    ax.set_ylim(-0.3, 2.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("CWL Inference")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    if gate_status is not None:
        _draw_gate_lines(ax, gate_status, t0_ms)


def _plot_racing_commands(ax, commands, gate_status, t0_ms):
    if commands is None:
        _no_data_placeholder(ax, "Control Inputs")
        return

    cmd = commands[commands["timestamp"] > 0].copy()
    if cmd.empty:
        _no_data_placeholder(ax, "Control Inputs")
        return

    t = (cmd["timestamp"] - t0_ms) / 1000.0

    for col, (color, label) in COMMAND_INPUTS.items():
        if col in cmd.columns:
            ax.plot(t, cmd[col], color=color, linewidth=0.8, alpha=0.8, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Input Rate")
    ax.set_title("Control Inputs")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.grid(alpha=0.3)

    if gate_status is not None:
        _draw_gate_lines(ax, gate_status, t0_ms)


def _plot_racing_command_limits(axes, commands, gate_status, t0_ms):
    """2×3 small-multiple panels: per-parameter flight profile limit over time.

    For each parameter the theoretical limit is linearly interpolated between
    FLIGHT_PROFILE_MIN_LIMITS (step 0) and FLIGHT_PROFILE_LIMITS (step N-1)
    using cwl_current_step from command_data.csv.

    Three filled bands make the operating envelope immediately readable:
      • green  — always-on minimum base
      • blue   — range unlocked by the current CWL step
      • red    — remaining locked headroom
    """
    params = list(_FLIGHT_PROFILE_LIMITS.keys())
    flat_axes = [ax for row in axes for ax in row]

    no_data = commands is None
    if not no_data:
        cmd = commands[commands["timestamp"] > 0].copy()
        no_data = (
            cmd.empty
            or "cwl_current_step" not in cmd.columns
            or "cwl_total_steps" not in cmd.columns
        )

    if no_data:
        for ax, param in zip(flat_axes, params):
            _no_data_placeholder(ax, _FLIGHT_PARAM_LABELS.get(param, (param, ""))[0])
        return

    cmd = cmd.sort_values("timestamp")
    t = (cmd["timestamp"] - t0_ms) / 1000.0
    steps = cmd["cwl_current_step"].values
    totals = cmd["cwl_total_steps"].replace(0, np.nan).fillna(24).values
    ratio = np.clip(steps / np.maximum(totals - 1, 1), 0.0, 1.0)

    for ax, param in zip(flat_axes, params):
        vmin = _FLIGHT_PROFILE_MIN_LIMITS.get(param, 0.0)
        vmax = _FLIGHT_PROFILE_LIMITS[param]
        label, unit = _FLIGHT_PARAM_LABELS.get(param, (param, ""))
        limit_vals = vmin + (vmax - vmin) * ratio

        ax.fill_between(t, 0, vmin, alpha=0.18, color="#4CAF50", linewidth=0)
        ax.fill_between(t, vmin, limit_vals, alpha=0.32, color="#3949AB", linewidth=0)
        ax.fill_between(t, limit_vals, vmax, alpha=0.10, color="#F44336", linewidth=0)
        ax.plot(t, limit_vals, color="#3949AB", linewidth=1.5, label="Current limit")
        ax.axhline(
            vmin, color="#4CAF50", linewidth=1.0, linestyle="--", alpha=0.9,
            label=f"Min ({vmin})",
        )
        ax.axhline(
            vmax, color="#F44336", linewidth=1.0, linestyle="--", alpha=0.9,
            label=f"Max ({vmax})",
        )

        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_ylabel(unit, fontsize=8)
        ax.set_ylim(min(0, vmin * 0.9), vmax * 1.08)
        ax.legend(fontsize=7, loc="upper left", handlelength=1.2)
        ax.grid(linestyle=":", alpha=0.3)

        if gate_status is not None:
            _draw_gate_lines(ax, gate_status, t0_ms)


def _plot_racing_adaptation(ax, commands, inference, t0_ms):
    if commands is None:
        _no_data_placeholder(ax, "Adaptation Parameters")
        return

    cmd = commands[commands["timestamp"] > 0].copy()
    available = [col for col in ADAPTATION_PARAMS if col in cmd.columns]

    if not available or cmd.empty:
        _no_data_placeholder(ax, "Adaptation Parameters")
        return

    t = (cmd["timestamp"] - t0_ms) / 1000.0

    # CWL background shading
    if inference is not None and len(inference) > 1:
        inf = inference[inference["timestamp"] > 0].copy()
        if not inf.empty:
            inf_t = (inf["timestamp"] - t0_ms) / 1000.0
            for i in range(len(inf) - 1):
                level = int(inf.iloc[i]["filtered_state"])
                ax.axvspan(
                    inf_t.iloc[i],
                    inf_t.iloc[i + 1],
                    alpha=0.08,
                    color=STATE_COLORS.get(level, "#999"),
                    linewidth=0,
                )

    for col in available:
        label, _, _ = ADAPTATION_PARAMS[col]
        vals = cmd[col]
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax - vmin < 1e-9:
            normalized = pd.Series(0.5, index=vals.index)
        else:
            normalized = (vals - vmin) / (vmax - vmin)
        ax.plot(
            t,
            normalized,
            linewidth=1.2,
            label=f"{label} [{vmin:.2f}–{vmax:.2f}]",
            alpha=0.85,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized (0 = restricted, 1 = full)")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Adaptation Parameters — CWL background")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)


def _plot_racing_splits(ax, gates, gate_status):
    if gate_status is None:
        _no_data_placeholder(ax, "Gate Split Times")
        return

    passed = _gate_passage_times(gate_status)
    if len(passed) < 2:
        _no_data_placeholder(ax, "Gate Split Times")
        return

    gate_ids = passed["id"].values
    timestamps = passed["first_pass_timestamp"].values
    splits = np.diff(timestamps) / 1000.0

    gate_info = gates.set_index("id")
    colors = []
    for i in range(1, len(gate_ids)):
        gid = gate_ids[i]
        if DIFFICULTY_COL in gate_info.columns and gid in gate_info.index:
            hard = bool(gate_info.loc[gid, DIFFICULTY_COL])
            colors.append("#F44336" if hard else "#4CAF50")
        else:
            colors.append("#999")

    x = np.arange(len(splits))
    labels = [f"{gate_ids[i]}→{gate_ids[i + 1]}" for i in range(len(splits))]

    ax.bar(x, splits, color=colors, edgecolor="white", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Split Time (s)")
    ax.set_title("Gate Split Times")
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def _plot_racing_stats(ax, gates, gate_status, drones, centroid):
    ax.axis("off")

    lines = []

    if gate_status is not None:
        passed = _gate_passage_times(gate_status)
        if len(passed) >= 2:
            total_s = (
                passed["first_pass_timestamp"].max()
                - passed["first_pass_timestamp"].min()
            ) / 1000.0
            mm = int(total_s // 60)
            ss = total_s % 60
            lines.append(f"Completion time (first→last gate):  {mm}m {ss:.1f}s")

        n_passed = int((gate_status["pass_count"] > 0).sum())
        n_full = int((gate_status["pass_count"] >= SWARM_SIZE).sum())
        lines.append(f"Gates reached:                   {n_passed} / {len(gates)}")
        lines.append(f"Gates fully passed (9/9):        {n_full} / {len(gates)}")

    if centroid is not None and "n_alive" in centroid.columns:
        lines.append(
            f"Min drones alive:                "
            f"{centroid['n_alive'].min()} / {SWARM_SIZE}"
        )
        lines.append(
            f"Final drones alive:              "
            f"{centroid['n_alive'].iloc[-1]} / {SWARM_SIZE}"
        )

    if not lines:
        lines.append("No performance data available")

    ax.text(
        0.05,
        0.9,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8),
    )
    ax.set_title("Performance Summary", fontsize=11, fontweight="bold")


def _detect_racing_mode(data_dir: Path) -> str:
    """Return 'trial', 'subject', or 'experiment' for racing data."""
    if (data_dir / GATE_LAYOUT_FILE).exists():
        return "trial"
    if _SUBJECT_RE.match(data_dir.name):
        return "subject"
    return "experiment"


def run_racing(show: bool, output_dir: Path, data_dir: Path, traj_type: str = "inference", **_kw):
    racing_mode = _detect_racing_mode(data_dir)
    print(f"Loading racing data from: {data_dir}  [{racing_mode} mode]")
    output_dir.mkdir(parents=True, exist_ok=True)
    if racing_mode == "experiment":
        _run_racing_experiment(show, output_dir, data_dir)
    elif racing_mode == "subject":
        _run_racing_subject(show, output_dir, data_dir)
    else:
        _run_racing_trial(show, output_dir, data_dir, traj_type)


def _run_racing_trial(show: bool, output_dir: Path, data_dir: Path, traj_type: str = "inference"):

    gates = _load_csv(data_dir, GATE_LAYOUT_FILE)
    gate_status = _load_csv(data_dir, GATE_STATUS_FILE)
    inference = _load_csv(data_dir, INFERENCE_FILE_NAME)
    commands = _load_csv(data_dir, COMMAND_DATA_FILE)
    drones = _load_csv(data_dir, DRONE_FILE_NAME)

    if gates is None:
        print("  ERROR: gate_layout.csv not found")
        return

    centroid = _compute_centroid(drones) if drones is not None else None

    # Build trajectory merged dataset (inference or adaptive coloring)
    traj_merged = None
    if centroid is not None:
        if traj_type == "adaptive" and commands is not None:
            cmd_cols = commands[
                ["timestamp", "cwl_current_step", "cwl_total_steps"]
            ].sort_values("timestamp")
            traj_merged = pd.merge_asof(
                centroid.sort_values("timestamp"),
                cmd_cols,
                on="timestamp",
                direction="backward",
            ).dropna(subset=["cwl_current_step"])
        elif inference is not None and len(inference) > 1:
            inf_cols = inference[["timestamp", "filtered_state", "raw_state"]].sort_values("timestamp")
            traj_merged = pd.merge_asof(
                centroid.sort_values("timestamp"),
                inf_cols,
                on="timestamp",
                direction="backward",
            ).dropna(subset=["filtered_state"])

    gate_breakdown = _compute_gate_breakdown(gates, gate_status, drones)
    dead_drones = _find_dead_drones(drones) if drones is not None else pd.DataFrame()
    t0_ms = _find_t0(inference, centroid, commands, gate_status)
    figs: list[tuple[plt.Figure, Path]] = []

    # ── Figure 1: Course Analysis ──
    fig1 = plt.figure(figsize=(18, 16))
    gs1 = fig1.add_gridspec(
        4,
        1,
        height_ratios=[3, 1.8, 1.2, 1.2],
        hspace=0.10,
    )
    fig1.suptitle("Racing Trial — Course Analysis", fontsize=14, fontweight="bold")

    ax_traj = fig1.add_subplot(gs1[0])
    ax_prob = fig1.add_subplot(gs1[1], sharex=ax_traj)
    ax_step = fig1.add_subplot(gs1[2], sharex=ax_traj)
    ax_miss = fig1.add_subplot(gs1[3], sharex=ax_traj)

    _plot_racing_combined(
        ax_traj, gates, centroid, traj_merged, gate_breakdown,
        dead_drones=dead_drones, traj_type=traj_type, gate_status=gate_status,
    )
    _plot_racing_cwl_probability(ax_prob, inference, centroid, gates)
    _plot_racing_adaptation_z(ax_step, commands, centroid, gates)
    _plot_racing_drone_misses(ax_miss, gates, gate_breakdown)

    for ax in (ax_traj, ax_prob, ax_step):
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("")

    fig1.tight_layout()
    figs.append((fig1, output_dir / "racing_course_analysis.png"))

    # ── Figure 2: Control Limits & Performance ──
    fig2 = plt.figure(figsize=(16, 13))
    fig2.suptitle(
        "Racing Trial — Control Limits & Performance", fontsize=14, fontweight="bold"
    )
    outer_gs = fig2.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.45)
    limits_gs = outer_gs[0].subgridspec(2, 3, hspace=0.55, wspace=0.38)
    bottom_gs = outer_gs[1].subgridspec(1, 2, wspace=0.35)

    # Build 2×3 limit axes — share x-axis within each column
    axes_limits = []
    for r in range(2):
        row = []
        for c in range(3):
            if r == 0:
                ax = fig2.add_subplot(limits_gs[r, c])
            else:
                ax = fig2.add_subplot(limits_gs[r, c], sharex=axes_limits[0][c])
            row.append(ax)
        axes_limits.append(row)
    for ax in axes_limits[0]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("")
    for ax in axes_limits[1]:
        ax.set_xlabel("Time (s)", fontsize=8)

    ax_splits = fig2.add_subplot(bottom_gs[0])
    ax_stats = fig2.add_subplot(bottom_gs[1])

    _plot_racing_command_limits(axes_limits, commands, gate_status, t0_ms)
    _plot_racing_splits(ax_splits, gates, gate_status)
    _plot_racing_stats(ax_stats, gates, gate_status, drones, centroid)
    fig2.tight_layout()
    figs.append((fig2, output_dir / "racing_adaptation_performance.png"))

    _save_or_show(figs, show)


# ── Per-subject racing analysis ──────────────────────────────────────────────


def _load_racing_trials(subject_dir: Path) -> list[dict]:
    """Load all racing trial folders under a subject directory.

    Returns one dict per trial with: name, gates, gate_status, inference,
    commands, drones, centroid, t0_ms.  Trials missing gate_layout or
    gate_status are skipped.
    """
    trial_dirs = sorted(
        d for d in subject_dir.iterdir()
        if d.is_dir() and d.name.startswith("trial_")
    )
    trials: list[dict] = []
    for trial_dir in trial_dirs:
        gates = _load_csv(trial_dir, GATE_LAYOUT_FILE)
        gate_status = _load_csv(trial_dir, GATE_STATUS_FILE)
        if gates is None or gate_status is None:
            continue
        inference = _load_csv(trial_dir, INFERENCE_FILE_NAME)
        commands = _load_csv(trial_dir, COMMAND_DATA_FILE)
        drones = _load_csv(trial_dir, DRONE_FILE_NAME)
        centroid = _compute_centroid(drones) if drones is not None else None
        t0_ms = _find_t0(inference, centroid, commands, gate_status)
        trials.append({
            "name": trial_dir.name,
            "gates": gates,
            "gate_status": gate_status,
            "inference": inference,
            "commands": commands,
            "drones": drones,
            "centroid": centroid,
            "t0_ms": t0_ms,
        })
    return trials


def _segment_metadata(
    gates: pd.DataFrame,
) -> tuple[list[int], list[int], list[str], list[bool]]:
    """Return canonical per-segment (gate_a_id, gate_b_id, name, is_hard).

    Gates are ordered by center_z (course progress).  A segment's difficulty
    is taken from the destination gate's `is_hard` column when present.
    Names use a per-difficulty counter, e.g. "Easy 1", "Hard 1", "Easy 2", ...
    """
    sorted_gates = gates.sort_values("center_z").reset_index(drop=True)
    has_diff = DIFFICULTY_COL in sorted_gates.columns

    a_ids: list[int] = []
    b_ids: list[int] = []
    names: list[str] = []
    diffs: list[bool] = []
    easy_n = 0
    hard_n = 0
    for i in range(len(sorted_gates) - 1):
        a_ids.append(int(sorted_gates.iloc[i]["id"]))
        b_ids.append(int(sorted_gates.iloc[i + 1]["id"]))
        hard = bool(sorted_gates.iloc[i + 1][DIFFICULTY_COL]) if has_diff else False
        diffs.append(hard)
        if hard:
            hard_n += 1
            names.append(f"Hard {hard_n}")
        else:
            easy_n += 1
            names.append(f"Easy {easy_n}")
    return a_ids, b_ids, names, diffs


def _plot_subject_completion_times(ax, trials):
    """Horizontal stacked bars per trial, segments split at gates.

    Easy/hard segments use distinct hues (green/red) and alternate between two
    shades of that hue so adjacent segment boundaries are clearly visible.
    Per-segment split times are annotated inside each segment when wide enough,
    and total completion time (mm:ss) is annotated at the right of each bar.
    """
    if not trials:
        _no_data_placeholder(ax, "Trial Completion Times")
        return

    easy_shades = ["#43A047", "#A5D6A7"]
    hard_shades = ["#E53935", "#FFCDD2"]

    y_positions = np.arange(len(trials))
    bar_height = 0.62
    max_total = 0.0
    drew_easy = False
    drew_hard = False

    for yi, tr in zip(y_positions, trials):
        gates = tr["gates"]
        passed = _gate_passage_times(tr["gate_status"])
        if len(passed) < 2 or gates is None:
            continue

        gates_idx = gates.set_index("id")
        ts = passed["first_pass_timestamp"].values / 1000.0
        ids = passed["id"].values
        ts0 = ts[0]

        easy_alt = 0
        hard_alt = 0
        for i in range(1, len(ts)):
            seg_start = ts[i - 1] - ts0
            seg_dur = ts[i] - ts[i - 1]
            if seg_dur <= 0:
                continue

            dest_id = int(ids[i])
            hard = (
                bool(gates_idx.loc[dest_id, DIFFICULTY_COL])
                if (
                    DIFFICULTY_COL in gates_idx.columns
                    and dest_id in gates_idx.index
                )
                else False
            )
            if hard:
                color = hard_shades[hard_alt % 2]
                hard_alt += 1
                drew_hard = True
            else:
                color = easy_shades[easy_alt % 2]
                easy_alt += 1
                drew_easy = True

            ax.barh(
                yi, seg_dur, left=seg_start, height=bar_height,
                color=color, edgecolor="white", linewidth=1.2, zorder=2,
            )

            if seg_dur >= 1.5:
                ax.text(
                    seg_start + seg_dur / 2, yi,
                    f"{seg_dur:.1f}s",
                    ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold", zorder=3,
                )

        total = ts[-1] - ts0
        max_total = max(max_total, total)
        mm = int(total // 60)
        ss = total % 60
        ax.text(
            total + 0.5, yi,
            f"{mm}m {ss:.1f}s",
            ha="left", va="center",
            fontsize=9, fontweight="bold", color="#212121",
        )

    if max_total <= 0:
        _no_data_placeholder(ax, "Trial Completion Times")
        return

    ax.set_yticks(y_positions)
    ax.set_yticklabels([tr["name"] for tr in trials], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, max_total * 1.18)
    ax.set_title(
        "Trial Completion Times — gate splits stacked, hue = difficulty"
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    from matplotlib.patches import Patch
    handles = []
    if drew_easy:
        handles.append(Patch(facecolor=easy_shades[0], edgecolor="white", label="Easy segment"))
    if drew_hard:
        handles.append(Patch(facecolor=hard_shades[0], edgecolor="white", label="Hard segment"))
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=8)


def _plot_subject_segment_cwl(ax, trials):
    """Per-segment CWL stripes — y groups course segments, sub-rows per trial.

    Each course segment (between consecutive gates ordered by center_z) becomes
    a horizontal band on the y-axis.  Within each band, each trial gets its own
    horizontal stripe colored by `filtered_state` (Low / Medium / High).  The
    CWL signal is time-normalized within the segment so all trials are
    horizontally aligned regardless of completion speed.

    Difficulty is shown as a soft background band (green = easy, red = hard).
    """
    if not trials:
        _no_data_placeholder(ax, "Segment CWL Projection")
        return

    gates = trials[0]["gates"]
    if gates is None or len(gates) < 2:
        _no_data_placeholder(ax, "Segment CWL Projection")
        return

    a_ids, b_ids, seg_names, seg_diffs = _segment_metadata(gates)
    n_segments = len(a_ids)
    if n_segments == 0:
        _no_data_placeholder(ax, "Segment CWL Projection")
        return

    n_trials = len(trials)
    seg_block = 1.0
    seg_gap = 0.3
    seg_pitch = seg_block + seg_gap
    line_pitch = seg_block / (n_trials + 1)

    from matplotlib.collections import LineCollection as _LC
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    y_centers: list[float] = []
    drew_any = False
    for s_idx in range(n_segments):
        seg_top = s_idx * seg_pitch
        y_centers.append(seg_top + seg_block / 2)

        bg_color = "#FFEBEE" if seg_diffs[s_idx] else "#E8F5E9"
        ax.axhspan(seg_top, seg_top + seg_block, color=bg_color, alpha=0.55, zorder=0)

        ga, gb = a_ids[s_idx], b_ids[s_idx]
        for t_idx, tr in enumerate(trials):
            inf = tr["inference"]
            gs = tr["gate_status"]
            if inf is None or gs is None:
                continue

            try:
                gs_idx = gs.set_index("id")
                t_start = float(gs_idx.loc[ga, "first_pass_timestamp"])
                t_end = float(gs_idx.loc[gb, "first_pass_timestamp"])
            except KeyError:
                continue
            if t_start <= 0 or t_end <= 0 or t_end <= t_start:
                continue

            mask = (inf["timestamp"] >= t_start) & (inf["timestamp"] <= t_end)
            inf_seg = inf.loc[mask].sort_values("timestamp")
            if len(inf_seg) < 2:
                continue

            x_norm = (inf_seg["timestamp"].values - t_start) / (t_end - t_start)
            cwl = inf_seg["filtered_state"].fillna(0).values.astype(int)

            y_line = seg_top + (t_idx + 1) * line_pitch
            points = np.column_stack([x_norm, np.full_like(x_norm, y_line)])
            segs = np.stack([points[:-1], points[1:]], axis=1)
            colors = [STATE_COLORS.get(int(c), "#999") for c in cwl[:-1]]
            ax.add_collection(
                _LC(segs, colors=colors, linewidths=8, capstyle="butt", zorder=3)
            )
            drew_any = True

            if s_idx == 0:
                ax.text(
                    -0.015, y_line, tr["name"].replace("trial_", "T"),
                    ha="right", va="center", fontsize=8, color="#444",
                )

    if not drew_any:
        _no_data_placeholder(ax, "Segment CWL Projection")
        return

    ax.set_yticks(y_centers)
    ax.set_yticklabels(seg_names, fontsize=10, fontweight="bold")
    ax.set_xlim(-0.06, 1.02)
    ax.set_ylim(n_segments * seg_pitch - seg_gap, -0.05)
    ax.set_xlabel("Normalized segment progress  (0 = entry → 1 = exit)")
    ax.set_title(
        "CWL Profile per Course Segment — one stripe per trial, time-normalized"
    )
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    handles = [
        Line2D([0], [0], color=STATE_COLORS[i], linewidth=6, label=f"CWL {STATE_LABELS[i]}")
        for i in range(3)
    ]
    handles += [
        Patch(facecolor="#E8F5E9", edgecolor="none", label="Easy bg"),
        Patch(facecolor="#FFEBEE", edgecolor="none", label="Hard bg"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=5)


def _run_racing_subject(show: bool, output_dir: Path, data_dir: Path):
    trials = _load_racing_trials(data_dir)
    if not trials:
        print("  No trial folders with gate_layout + gate_status found.")
        return
    print(f"  Loaded {len(trials)} trial(s)")

    figs: list[tuple[plt.Figure, Path]] = []
    subject = data_dir.name

    fig1_h = max(3.5, 0.9 + 0.7 * len(trials))
    fig1, ax1 = plt.subplots(figsize=(15, fig1_h))
    fig1.suptitle(
        f"Racing — {subject} — Trial Completion Times",
        fontsize=13, fontweight="bold",
    )
    _plot_subject_completion_times(ax1, trials)
    fig1.tight_layout()
    figs.append((fig1, output_dir / f"racing_subject_{subject}_completion_times.png"))

    # Height grows with both segments and trials
    n_segs_est = max(1, len(trials[0]["gates"]) - 1) if trials[0]["gates"] is not None else 5
    fig2_h = max(6.5, 0.7 * n_segs_est + 0.25 * n_segs_est * len(trials))
    fig2, ax2 = plt.subplots(figsize=(15, min(fig2_h, 14)))
    fig2.suptitle(
        f"Racing — {subject} — Segment CWL Profile",
        fontsize=13, fontweight="bold",
    )
    _plot_subject_segment_cwl(ax2, trials)
    fig2.tight_layout()
    figs.append((fig2, output_dir / f"racing_subject_{subject}_segment_cwl.png"))

    _save_or_show(figs, show)


# ── Per-experiment racing analysis ───────────────────────────────────────────


def _classify_subject_adaptation(trials: list[dict]) -> str:
    """Heuristic group classifier: 'adaptive' / 'non_adaptive' / 'unknown'.

    A subject is 'adaptive' if cwl_current_step varies across any of their
    trials — i.e. the system actually moved between flight profiles.
    A constant step across all trials means the profile was pinned (control
    group). Subjects with no command data fall back to 'unknown'.
    """
    saw_data = False
    for tr in trials:
        commands = tr["commands"]
        if commands is None or "cwl_current_step" not in commands.columns:
            continue
        steps = commands["cwl_current_step"].dropna()
        if steps.empty:
            continue
        saw_data = True
        if steps.nunique() > 1:
            return "adaptive"
    return "non_adaptive" if saw_data else "unknown"


def _trial_metrics(tr: dict) -> dict | None:
    """Compute summary metrics for one trial.

    completion_s   — first → last gate in seconds.
    min_alive      — minimum centroid n_alive over the trial.
    final_alive    — n_alive at the last centroid sample.
    dead_drones    — SWARM_SIZE − final_alive.
    missed_drones  — Σ over reached gates of (alive − inside) at gate time.
                     Counts every "alive drone outside the gate" event, so
                     a chronic straggler is penalised at every gate it skips.
    penalty_s      — completion + dead·DEAD_PENALTY + miss·MISS_PENALTY.
    """
    gates = tr["gates"]
    gs = tr["gate_status"]
    drones = tr["drones"]
    centroid = tr["centroid"]
    if gates is None or gs is None:
        return None

    passed = _gate_passage_times(gs)
    if len(passed) < 2:
        return None

    completion_s = (
        passed["first_pass_timestamp"].max()
        - passed["first_pass_timestamp"].min()
    ) / 1000.0

    breakdown = _compute_gate_breakdown(gates, gs, drones)
    missed_drones = sum(
        b.get("outside", 0) for b in breakdown.values() if b.get("reached")
    )

    if centroid is not None and "n_alive" in centroid.columns:
        min_alive = int(centroid["n_alive"].min())
        final_alive = int(centroid["n_alive"].iloc[-1])
    else:
        min_alive = SWARM_SIZE
        final_alive = SWARM_SIZE
    dead_drones = SWARM_SIZE - final_alive

    gates_reached = (
        int((gs["pass_count"] > 0).sum())
        if "pass_count" in gs.columns
        else len(passed)
    )

    penalty_s = (
        completion_s
        + dead_drones * RACING_DEAD_PENALTY_S
        + missed_drones * RACING_MISS_PENALTY_S
    )

    return {
        "completion_s": completion_s,
        "min_alive": min_alive,
        "final_alive": final_alive,
        "dead_drones": dead_drones,
        "missed_drones": missed_drones,
        "gates_reached": gates_reached,
        "gates_total": len(gates),
        "penalty_s": penalty_s,
    }


def _load_experiment_racing(experiment_dir: Path) -> dict[str, list[dict]]:
    """Load racing trials for every 4-char subject folder under experiment_dir."""
    by_subject: dict[str, list[dict]] = {}
    for subj_dir in sorted(experiment_dir.iterdir()):
        if not subj_dir.is_dir() or not _SUBJECT_RE.match(subj_dir.name):
            continue
        trials = _load_racing_trials(subj_dir)
        if trials:
            by_subject[subj_dir.name] = trials
    return by_subject


def _build_experiment_metrics(
    by_subject: dict[str, list[dict]],
    groups: dict[str, str],
) -> pd.DataFrame:
    """Long DataFrame: one row per (subject, trial) with metrics + group."""
    rows = []
    for sid, trials in by_subject.items():
        group = groups.get(sid, "unknown")
        for tr in trials:
            m = _trial_metrics(tr)
            if m is None:
                continue
            rows.append({
                "subject_id": sid,
                "trial": tr["name"],
                "group": group,
                **m,
            })
    return pd.DataFrame(rows)


def _grouped_boxplot(
    ax,
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    better_low: bool,
):
    """Per-subject boxes coloured by adaptation group, with per-trial dots.

    Subjects are ordered: adaptive → non_adaptive → unknown, alphabetical
    within group. A dashed horizontal line marks each group's mean across
    its trials and is annotated with μ. A vertical dotted line separates
    groups for readability.
    """
    if df.empty or value_col not in df.columns:
        _no_data_placeholder(ax, title)
        return

    groups_order = ["adaptive", "non_adaptive", "unknown"]
    box_data: list[np.ndarray] = []
    box_colors: list[str] = []
    box_labels: list[str] = []
    box_groups: list[str] = []

    for g in groups_order:
        sids = sorted(df[df["group"] == g]["subject_id"].unique())
        for sid in sids:
            vals = df[df["subject_id"] == sid][value_col].dropna().values
            if len(vals) == 0:
                continue
            box_data.append(vals)
            box_colors.append(GROUP_COLORS[g])
            box_labels.append(sid)
            box_groups.append(g)

    if not box_data:
        _no_data_placeholder(ax, title)
        return

    positions = np.arange(1, len(box_data) + 1)
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.6},
        flierprops={"marker": ".", "markersize": 4, "alpha": 0.5},
    )
    import matplotlib.colors as _mcolors
    for patch, c in zip(bp["boxes"], box_colors, strict=True):
        rgba = (*_mcolors.to_rgb(c), 0.55)
        patch.set_facecolor(rgba)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.2)

    rng = np.random.RandomState(42)
    for pos, data, c in zip(positions, box_data, box_colors, strict=True):
        jitter = (rng.rand(len(data)) - 0.5) * 0.28
        ax.scatter(
            pos + jitter, data, color=c, alpha=0.85, s=20,
            edgecolors="white", linewidths=0.5, zorder=4,
        )

    # Group means + group separators
    last_g = None
    for i, (pos, g) in enumerate(zip(positions, box_groups)):
        if last_g is not None and g != last_g:
            ax.axvline(pos - 0.5, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)
        last_g = g

    for g in groups_order:
        gdata = df[df["group"] == g][value_col].dropna().values
        if len(gdata) == 0:
            continue
        gxs = [pos for pos, gg in zip(positions, box_groups) if gg == g]
        if not gxs:
            continue
        gmean = float(np.mean(gdata))
        x0, x1 = min(gxs) - 0.45, max(gxs) + 0.45
        ax.hlines(gmean, x0, x1, colors=GROUP_COLORS[g], linestyles="--",
                  linewidth=1.6, alpha=0.95, zorder=2)
        ax.text(
            (x0 + x1) / 2, gmean,
            f" {GROUP_LABELS[g].split()[0]} μ={gmean:.2f} ",
            ha="center", va="bottom", fontsize=8, color=GROUP_COLORS[g],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=GROUP_COLORS[g], linewidth=0.7, alpha=0.85),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(box_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    indicator = "↓ better" if better_low else "↑ better"
    ax.text(
        0.01, 0.97, indicator, transform=ax.transAxes,
        fontsize=8, va="top", ha="left", color="#444",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor="#ccc", linewidth=0.6, alpha=0.85),
    )


def _plot_experiment_summary_boxplots(axes, df: pd.DataFrame):
    """4-panel grid: completion, min alive, missed, penalty."""
    panels = [
        ("completion_s", "Completion time (s)",
         "Trial Completion Time", True),
        ("min_alive", f"Min drones alive (/{SWARM_SIZE})",
         "Min Drones Alive per Trial", False),
        ("missed_drones", "Σ drones outside gates",
         "Drone-Gate Misses per Trial", True),
        ("penalty_s", "Penalty score (s-equivalent)",
         f"Composite Penalty\n(time + {RACING_DEAD_PENALTY_S:.0f}s/dead "
         f"+ {RACING_MISS_PENALTY_S:.0f}s/miss)", True),
    ]
    flat = np.asarray(axes).ravel()
    for ax, (col, ylabel, title, low_better) in zip(flat, panels):
        _grouped_boxplot(ax, df, col, ylabel, title, better_low=low_better)


def _plot_experiment_cwl_distribution(
    ax, by_subject: dict[str, list[dict]], groups: dict[str, str],
):
    """Stacked horizontal bars: % time in Low/Med/High CWL per subject."""
    rows = []
    for sid, trials in by_subject.items():
        all_states: list[int] = []
        for tr in trials:
            inf = tr["inference"]
            if inf is None or "filtered_state" not in inf.columns:
                continue
            all_states.extend(inf["filtered_state"].dropna().astype(int).tolist())
        if not all_states:
            continue
        s = np.array(all_states)
        total = len(s)
        rows.append({
            "subject_id": sid,
            "group": groups.get(sid, "unknown"),
            "low_pct": float((s == 0).sum()) / total,
            "med_pct": float((s == 1).sum()) / total,
            "high_pct": float((s == 2).sum()) / total,
        })

    if not rows:
        _no_data_placeholder(ax, "CWL Distribution")
        return

    df = pd.DataFrame(rows)
    order = {"adaptive": 0, "non_adaptive": 1, "unknown": 2}
    df["g_ord"] = df["group"].map(order).fillna(3)
    df = df.sort_values(["g_ord", "subject_id"]).reset_index(drop=True)

    y = np.arange(len(df))
    ax.barh(y, df["low_pct"], color=STATE_COLORS[0], label=STATE_LABELS[0],
            edgecolor="white", linewidth=0.6)
    ax.barh(y, df["med_pct"], left=df["low_pct"], color=STATE_COLORS[1],
            label=STATE_LABELS[1], edgecolor="white", linewidth=0.6)
    ax.barh(y, df["high_pct"], left=df["low_pct"] + df["med_pct"],
            color=STATE_COLORS[2], label=STATE_LABELS[2],
            edgecolor="white", linewidth=0.6)

    labels = [
        f"{r.subject_id} [{GROUP_LABELS[r.group].split()[0]}]"
        for r in df.itertuples()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Time fraction in CWL state")
    ax.set_title("CWL Distribution per Subject", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=3)


def _plot_experiment_adaptation_steps(
    ax, by_subject: dict[str, list[dict]], groups: dict[str, str],
):
    """Box plot of cwl_current_step per adaptive subject + grand mean line.

    Skipped for non_adaptive subjects since their step is constant by
    construction. The grand mean across all adaptive trials is a useful
    reference for picking constant-step values for new control subjects.
    """
    rows = []
    for sid, trials in by_subject.items():
        if groups.get(sid) != "adaptive":
            continue
        steps: list[float] = []
        for tr in trials:
            commands = tr["commands"]
            if commands is None or "cwl_current_step" not in commands.columns:
                continue
            steps.extend(commands["cwl_current_step"].dropna().tolist())
        if steps:
            rows.append({"subject_id": sid, "steps": np.array(steps)})

    if not rows:
        _no_data_placeholder(ax, "Adaptation Step Distribution")
        return

    rows.sort(key=lambda r: r["subject_id"])
    box_data = [r["steps"] for r in rows]
    positions = np.arange(1, len(rows) + 1)

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.6},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
    )
    import matplotlib.colors as _mcolors
    color = GROUP_COLORS["adaptive"]
    rgba = (*_mcolors.to_rgb(color), 0.55)
    for patch in bp["boxes"]:
        patch.set_facecolor(rgba)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

    grand_mean = float(np.concatenate(box_data).mean())
    ax.axhline(
        grand_mean, color="black", linewidth=1.2, linestyle="--",
        label=f"Adaptive grand μ = {grand_mean:.1f}",
    )

    # Indicate the experiment's max step from the first adaptive trial we find
    max_step = None
    for sid, trials in by_subject.items():
        if groups.get(sid) != "adaptive":
            continue
        for tr in trials:
            commands = tr["commands"]
            if commands is None or "cwl_total_steps" not in commands.columns:
                continue
            tot = commands["cwl_total_steps"].dropna()
            if not tot.empty:
                max_step = int(tot.iloc[0]) - 1
                break
        if max_step is not None:
            break
    if max_step is not None:
        ax.axhline(
            max_step, color="#888", linewidth=0.8, linestyle=":",
            label=f"Max step ({max_step})",
        )
        ax.axhline(
            0, color="#888", linewidth=0.8, linestyle=":", label="Min step (0)",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [r["subject_id"] for r in rows], rotation=30, ha="right", fontsize=8,
    )
    ax.set_ylabel("CWL current step")
    ax.set_title(
        "Adaptation Step Distribution per Subject (adaptive group)",
        fontsize=10, fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)


def _print_experiment_group_summary(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("GROUP SUMMARY — mean ± std across all trials")
    print("=" * 70)
    cols = [
        ("completion_s", "Completion time (s)", "{:>7.1f}"),
        ("min_alive",    f"Min alive (/{SWARM_SIZE})", "{:>7.2f}"),
        ("missed_drones", "Drone-gate misses",  "{:>7.2f}"),
        ("dead_drones",  f"Dead (/{SWARM_SIZE})", "{:>7.2f}"),
        ("penalty_s",    "Penalty (s)",         "{:>7.1f}"),
    ]
    for group in ["adaptive", "non_adaptive", "unknown"]:
        sub = df[df["group"] == group]
        if sub.empty:
            continue
        n_subj = sub["subject_id"].nunique()
        print(
            f"\n  {GROUP_LABELS[group]}  "
            f"(n={len(sub)} trials, {n_subj} subject(s))"
        )
        for col, label, fmt in cols:
            mu = sub[col].mean()
            sd = sub[col].std()
            print(f"    {label:<22}  {fmt.format(mu)}  ± {fmt.format(sd).strip()}")


def _run_racing_experiment(show: bool, output_dir: Path, data_dir: Path):
    by_subject = _load_experiment_racing(data_dir)
    if not by_subject:
        print("  No subject folders with racing trials found.")
        return

    groups = {
        sid: _classify_subject_adaptation(trials)
        for sid, trials in by_subject.items()
    }
    n_total_trials = sum(len(t) for t in by_subject.values())
    print(f"  Loaded {len(by_subject)} subject(s), {n_total_trials} trial(s)")
    print("\n  Group classification (auto-detected from cwl_current_step variance):")
    for sid in sorted(by_subject):
        print(f"    {sid}  →  {GROUP_LABELS[groups[sid]]}")

    df = _build_experiment_metrics(by_subject, groups)
    if df.empty:
        print("  No trial metrics could be computed.")
        return

    _print_experiment_group_summary(df)

    figs: list[tuple[plt.Figure, Path]] = []

    # Figure 1: 4-panel performance box plots
    fig1, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig1.suptitle(
        f"Racing Experiment — Performance Summary "
        f"({df['subject_id'].nunique()} subjects, {len(df)} trials)",
        fontsize=14, fontweight="bold",
    )
    _plot_experiment_summary_boxplots(axes, df)
    fig1.tight_layout()
    figs.append((fig1, output_dir / "racing_experiment_performance.png"))

    # Figure 2: CWL & adaptation profile
    fig2_h = max(5.5, 0.45 * len(by_subject) + 2.5)
    fig2, (ax_cwl, ax_step) = plt.subplots(
        1, 2, figsize=(16, min(fig2_h, 10)),
        gridspec_kw={"width_ratios": [1, 1.2]},
    )
    fig2.suptitle(
        "Racing Experiment — CWL & Adaptation Profile",
        fontsize=14, fontweight="bold",
    )
    _plot_experiment_cwl_distribution(ax_cwl, by_subject, groups)
    _plot_experiment_adaptation_steps(ax_step, by_subject, groups)
    fig2.tight_layout()
    figs.append((fig2, output_dir / "racing_experiment_cwl_adaptation.png"))

    _save_or_show(figs, show)


RESULT_TYPES = {
    "inference": run_inference,
    "racing": run_racing,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Supported result types: " + ", ".join(RESULT_TYPES),
    )
    parser.add_argument("result_type", choices=list(RESULT_TYPES))
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA, metavar="DIR")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, metavar="DIR")
    parser.add_argument(
        "--cwl",
        type=int,
        default=None,
        choices=[0, 1, 2],
        metavar="CWL",
        help="CWL level to visualize as a trajectory plot: 0=Low, 1=Medium, "
        "2=High.  The corresponding task is resolved automatically per subject.",
    )
    parser.add_argument(
        "--type",
        dest="traj_type",
        default="inference",
        choices=["inference", "adaptive"],
        help="Trajectory colormap for racing plot: 'inference' colors by CWL state "
        "(green/orange/red), 'adaptive' colors by adaptation step "
        "(red=Soft/0 → green=Racing/max).",
    )

    args = parser.parse_args()
    RESULT_TYPES[args.result_type](
        show=args.show, output_dir=args.output, data_dir=args.data,
        cwl=args.cwl, traj_type=args.traj_type,
    )


if __name__ == "__main__":
    main()
