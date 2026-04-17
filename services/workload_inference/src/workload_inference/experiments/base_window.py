"""Shared abstract base window for all experiment types."""

import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from workload_inference.constants import DATA_DIR
from workload_inference.experiments.base import ExperimentManager
from workload_inference.experiments.data_structures import (
    ExperimentState,
    ExperimentStatus,
)
from workload_inference.inference import InferenceSettings, WorkloadInferenceEngine

logger = logging.getLogger("ExperimentManagerWindow")


class ExperimentManagerWindow(QMainWindow):
    """
    Shared PyQt window for all experiment types.

    Provides the common layout:
      - Title bar
      - Header split vertically:
          Left (larger): pluggable experiment-specific info panel (abstract)
          Right:
            Upper: state boxes (Previous → Current → Next) + Next State button
            Lower: experiment name / subject UID / elapsed timer
      - Bottom canvas: gaze (left) | drone + workload (right)

    Subclasses must implement:
      - _create_drone_visualizer()
      - _initialize_experiment_info_panel()
      - _update_experiment_info_panel(status)
    """

    def __init__(self, experiment_manager: ExperimentManager):
        super().__init__()
        self.experiment_manager = experiment_manager
        self._is_status_error = True
        self._timer_auto_started = False

        self._initialize_core_components()
        self._initialize_experiment_info_panel()  # subclass hook
        self._initialize_shared_controls()
        self._initialize_canvas()

        self.experiment_manager.register_api_ready_listener(self.attach_listeners)

    # ── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def _create_drone_visualizer(self) -> Any:
        """Return the drone canvas widget for this experiment type."""
        ...

    @abstractmethod
    def _initialize_experiment_info_panel(self) -> None:
        """Build experiment-specific info widgets and add them to
        ``self._experiment_specific_panel_layout``."""
        ...

    @abstractmethod
    def _update_experiment_info_panel(self, status: ExperimentStatus) -> None:
        """Update the experiment-specific info labels. Called every 500 ms."""
        ...

    # ── Shared initialisation ────────────────────────────────────────────────

    def _initialize_core_components(self):
        self.setWindowTitle("Experiment Manager")
        self.setGeometry(100, 100, 1200, 800)
        self._layout = QVBoxLayout()
        self._central_widget = QWidget()
        self._central_widget.setLayout(self._layout)
        self.setCentralWidget(self._central_widget)

        # Title
        self._title_label = QLabel("Experiment Management")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self._layout.addWidget(self._title_label)

        # Header: left panel (experiment-specific, larger) | right panel
        header_widget = QWidget()
        header_layout = QHBoxLayout()
        header_widget.setLayout(header_layout)
        self._layout.addWidget(header_widget, 0)

        # Left panel — populated by subclass via _initialize_experiment_info_panel
        self._experiment_specific_panel = QWidget()
        self._experiment_specific_panel_layout = QVBoxLayout()
        self._experiment_specific_panel.setLayout(
            self._experiment_specific_panel_layout
        )
        header_layout.addWidget(self._experiment_specific_panel, 2)

        # Right panel — upper (states + button) and lower (meta + timer)
        self._right_panel = QWidget()
        self._right_panel_layout = QVBoxLayout()
        self._right_panel.setLayout(self._right_panel_layout)
        header_layout.addWidget(self._right_panel, 1)

    def _initialize_shared_controls(self):
        """Build the right panel: upper = states + Next State btn,
        lower = experiment name, subject UID, timer."""
        state_label_stylesheet = (
            "border: 2px solid black; padding: 20px; font-size: 16px;"
        )

        # ── Upper right: state boxes + Next State button ──────────────────────
        upper_right = QWidget()
        upper_right_layout = QHBoxLayout()
        upper_right.setLayout(upper_right_layout)
        self._right_panel_layout.addWidget(upper_right, 1)

        self._previous_state_label = QLabel("Previous")
        self._previous_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._previous_state_label.setStyleSheet(state_label_stylesheet)
        upper_right_layout.addWidget(self._previous_state_label, 1)

        arrow1 = QLabel("→")
        arrow1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow1.setStyleSheet("font-weight: bold; font-size: 32px;")
        upper_right_layout.addWidget(arrow1)

        self._current_state_label = QLabel("Current")
        self._current_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._current_state_label.setStyleSheet(state_label_stylesheet)
        upper_right_layout.addWidget(self._current_state_label, 1)

        arrow2 = QLabel("→")
        arrow2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow2.setStyleSheet("font-weight: bold; font-size: 32px;")
        upper_right_layout.addWidget(arrow2)

        self._next_state_label = QLabel("Next")
        self._next_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._next_state_label.setStyleSheet(state_label_stylesheet)
        upper_right_layout.addWidget(self._next_state_label, 1)

        self._next_state_btn = QPushButton("Next State")
        self._next_state_btn.setMinimumHeight(60)
        self._next_state_btn.clicked.connect(self.experiment_manager.request_next_state)
        self._next_state_btn.setEnabled(False)
        upper_right_layout.addWidget(self._next_state_btn, 1)

        # ── Lower right: experiment name, subject UID, task/trial, timer ─────
        lower_right = QWidget()
        lower_right_layout = QHBoxLayout()
        lower_right.setLayout(lower_right_layout)
        self._right_panel_layout.addWidget(lower_right, 1)

        exp_label = QLabel("Experiment:")
        exp_label.setStyleSheet("font-weight: bold;")
        lower_right_layout.addWidget(exp_label)

        self._experiment_name_value_label = QLabel(
            self.experiment_manager.experiment_config.get("name", "unknown")
        )
        lower_right_layout.addWidget(self._experiment_name_value_label)

        lower_right_layout.addWidget(self._make_separator())

        uid_label = QLabel("Subject ID:")
        uid_label.setStyleSheet("font-weight: bold;")
        lower_right_layout.addWidget(uid_label)

        self._uid_value_label = QLabel(
            self.experiment_manager.experiment_config.get("participant", {}).get(
                "uid", "????"
            )
        )
        lower_right_layout.addWidget(self._uid_value_label)

        lower_right_layout.addWidget(self._make_separator())

        self._task_number_value_label = QLabel("Task #0")
        lower_right_layout.addWidget(self._task_number_value_label)

        arrow_task = QLabel("→")
        arrow_task.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow_task.setStyleSheet("font-size: 18px; color: gray;")
        lower_right_layout.addWidget(arrow_task)

        self._trial_number_value_label = QLabel("Trial #0")
        lower_right_layout.addWidget(self._trial_number_value_label)

        lower_right_layout.addWidget(self._make_separator())

        timer_widget = QWidget()
        timer_layout = QVBoxLayout(timer_widget)
        timer_layout.setContentsMargins(0, 0, 0, 0)
        timer_layout.setSpacing(2)

        self._ellapsed_time_label = QLabel("00:00")
        self._ellapsed_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ellapsed_time_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        timer_layout.addWidget(self._ellapsed_time_label)

        self.start_ellapsed_time_button = QPushButton("Start timer")
        self.start_ellapsed_time_button.clicked.connect(self._start_experiment_timer)
        timer_layout.addWidget(self.start_ellapsed_time_button)

        lower_right_layout.addWidget(timer_widget)

        self._ellapsed_timer = QTimer()
        self._ellapsed_timer.timeout.connect(self._update_ellapsed_time)

    def _initialize_canvas(self):
        """Build the bottom canvas: gaze left, drone + workload right."""
        from workload_inference.visualize import (
            GazeDataCanvas,
            WorkloadDisplayWidget,
        )

        self._gaze_visualizer = GazeDataCanvas(
            parent=self,
            screen_width=1920,
            screen_height=1200,
            plotting_window=200,
        )
        self._drone_visualizer = self._create_drone_visualizer()

        def _resolve(p: str | None) -> str | None:
            if p is None:
                return None

            path = Path(p)
            return str(path if path.is_absolute() else DATA_DIR.parent / path)

        model_path = _resolve(
            self.experiment_manager.experiment_config.get("workload_model_path", None)
        )
        settings_path = Path(model_path).parent / "settings.yml" if model_path else None
        if settings_path is not None:
            try:
                settings = InferenceSettings.from_yaml(settings_path)
            except FileNotFoundError:
                logger.warning(
                    "Workload settings file '%s' not found, using defaults",
                    settings_path,
                )
                settings = InferenceSettings()
        else:
            settings = InferenceSettings()

        self._workload_engine = WorkloadInferenceEngine.create(
            model_path=model_path, settings=settings
        )
        self._workload_display = WorkloadDisplayWidget(
            parent=self, engine=self._workload_engine
        )

        canvas_widget = QWidget()
        canvas_layout = QHBoxLayout()
        canvas_widget.setLayout(canvas_layout)
        canvas_layout.addWidget(self._gaze_visualizer, 1)

        right_pane = QWidget()
        right_pane_layout = QVBoxLayout()
        right_pane_layout.addWidget(self._drone_visualizer, 1)
        right_pane_layout.addWidget(self._workload_display, 0)
        right_pane.setLayout(right_pane_layout)
        canvas_layout.addWidget(right_pane, 1)

        self._layout.addWidget(canvas_widget, 1)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_separator() -> QLabel:
        sep = QLabel("|")
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep.setStyleSheet("font-size: 18px; color: gray;")
        return sep

    # ── Public start ─────────────────────────────────────────────────────────

    def start(self):
        self._flash_visible = True
        self._experiment_status_update_timer = QTimer()
        self._experiment_status_update_timer.timeout.connect(
            self._update_experiment_status
        )
        self._experiment_status_update_timer.start(500)
        self.showMaximized()

    # ── Timer callbacks ───────────────────────────────────────────────────────

    def _update_ellapsed_time(self):
        if self.experiment_manager._duration is not None:
            self._ellapsed_timer.stop()
        if self.experiment_manager._start_time is None:
            self._ellapsed_time_label.setText("00:00")
            return
        elapsed_seconds = int(time.time() - self.experiment_manager._start_time)
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60
        self._ellapsed_time_label.setText(f"{minutes:02d}:{seconds:02d}")

    def _update_experiment_status(self):
        self._toggle_current_state_border()
        if self.experiment_manager.api_on_error:
            self._is_status_error = True
            self._next_state_btn.setEnabled(False)
            return

        status = self.experiment_manager.experiment_status
        if status is None:
            self._is_status_error = True
            self._next_state_btn.setEnabled(False)
            return

        self._is_status_error = False
        self._next_state_btn.setEnabled(True)

        self._previous_state_label.setText(status.previous_state.name or "None")
        self._current_state_label.setText(status.current_state.name or "None")
        self._next_state_label.setText(status.next_state.name or "None")
        self._task_number_value_label.setText(f"#{status.current_task}")
        self._trial_number_value_label.setText(f"#{status.current_trial}")

        if (
            not self._timer_auto_started
            and status.current_state == ExperimentState.Welcome
        ):
            self._start_experiment_timer()
            self._timer_auto_started = True

        self._update_experiment_info_panel(status)

    def _toggle_current_state_border(self):
        if self._flash_visible:
            if self._is_status_error:
                self._title_label.setStyleSheet(
                    "font-size: 24px; font-weight: bold; background-color: red;"
                )
                self._current_state_label.setStyleSheet(
                    "border: 2px solid red; padding: 20px; font-size: 16px;"
                )
            else:
                self._current_state_label.setStyleSheet(
                    "border: 2px solid green; padding: 20px; font-size: 16px;"
                )
        else:
            self._current_state_label.setStyleSheet(
                "border: 2px solid black; padding: 20px; font-size: 16px;"
            )
            self._title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self._flash_visible = not self._flash_visible

    def _start_experiment_timer(self):
        self.experiment_manager._start_time = time.time()
        if not self._ellapsed_timer.isActive():
            self._ellapsed_timer.start(1000)
        self.start_ellapsed_time_button.setText("Reset timer")

    # ── Listeners ─────────────────────────────────────────────────────────────

    def attach_listeners(self):
        if self.experiment_manager.gaze_receiver is not None:
            self.experiment_manager.gaze_receiver.register_listener(
                self._gaze_visualizer.datas_callback
            )
            self.experiment_manager.gaze_receiver.register_listener(
                self._workload_engine.gaze_datas_callback
            )
        else:
            logger.warning(
                "Gaze receiver is not initialized. Cannot attach gaze visualizer listener."
            )
        if self.experiment_manager.drone_receiver is not None:
            self.experiment_manager.drone_receiver.register_listener(
                self._drone_visualizer.datas_callback
            )
        else:
            logger.warning(
                "Drone receiver is not initialized. Cannot attach drone visualizer listener."
            )
        if self.experiment_manager.user_input_receiver is not None:
            self.experiment_manager.user_input_receiver.register_listener(
                self._workload_display.on_user_input_data
            )
        else:
            logger.warning(
                "User input receiver is not initialized. "
                "Cannot attach pilot profile listener."
            )
        self._workload_engine.register_listener(
            self.experiment_manager.inference_callback
        )

    # ── Close ─────────────────────────────────────────────────────────────────

    def closeEvent(self, event: Any) -> None:
        try:
            if self._experiment_status_update_timer:
                self._experiment_status_update_timer.stop()
            try:
                self.experiment_manager.close()
            except Exception:
                logger.exception("Error while stopping receivers during close")
        finally:
            event.accept()
