"""Data structures for workload inference experiments."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TypeVar

import numpy as np


# Type aliases
class DataclassLike(Protocol):
    @classmethod
    def size(cls) -> int: ...
    @classmethod
    def from_buffer(cls, data: bytes) -> DataclassLike: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataclassLike: ...


T = TypeVar("T", bound=DataclassLike)


class Listener(Protocol[T]):
    def __call__(self, datas: Sequence[T], batch_update: bool = False) -> None: ...


@dataclass
class Metadata:
    is_sender_ready: np.uint8
    calibration_ok: np.uint8
    is_receiver_ready: np.uint8
    head: np.int32
    tail: np.int32

    def get_conversion_str(self) -> str:
        return "BBBII"

    def __len__(self) -> int:
        return self.size()

    @classmethod
    def size(cls) -> int:
        return 1 + 1 + 1 + 4 + 4

    @classmethod
    def from_buffer(cls, buffer: bytes) -> Metadata:
        return Metadata(
            is_sender_ready=np.frombuffer(buffer[0:1], dtype=np.uint8)[0],
            calibration_ok=np.frombuffer(buffer[1:2], dtype=np.uint8)[0],
            is_receiver_ready=np.frombuffer(buffer[2:3], dtype=np.uint8)[0],
            head=np.frombuffer(buffer[3:7], dtype=np.int32)[0],
            tail=np.frombuffer(buffer[7:11], dtype=np.int32)[0],
        )


@dataclass
class NBackData:
    timestamp: np.int64
    response_timestamp: np.int64
    nback_level: np.int8
    stimulus: np.int8
    participant_response: np.int8
    is_correct: np.int8

    def get_conversion_str(self) -> str:
        return "<2q4B"

    def __len__(self) -> int:
        return self.size()

    @classmethod
    def size(cls) -> int:
        return 8 + 8 + 1 + 1 + 1 + 1

    @classmethod
    def from_buffer(cls, buffer: bytes) -> NBackData:
        if len(buffer) < cls.size():
            raise ValueError(
                f"Buffer size {len(buffer)} is smaller than expected size {cls.size()}."
            )
        return NBackData(
            timestamp=np.frombuffer(buffer[0:8], dtype=np.int64)[0],
            response_timestamp=np.frombuffer(buffer[8:16], dtype=np.int64)[0],
            nback_level=np.frombuffer(buffer[16:17], dtype=np.int8)[0],
            stimulus=np.frombuffer(buffer[17:18], dtype=np.int8)[0],
            participant_response=np.frombuffer(buffer[18:19], dtype=np.int8)[0],
            is_correct=np.frombuffer(buffer[19:20], dtype=np.int8)[0],
        )

    @classmethod
    def from_dict(cls, data: dict) -> NBackData:
        try:
            return NBackData(
                timestamp=np.int64(data["timestamp"]),
                response_timestamp=np.int64(data["response_timestamp"]),
                nback_level=np.int8(data["nback_level"]),
                stimulus=np.int8(data["stimulus"]),
                participant_response=np.int8(data["participant_response"]),
                is_correct=np.int8(data["is_correct"]),
            )
        except KeyError as e:
            raise ValueError(f"Missing key in data dictionary: {e}") from e


@dataclass
class DroneData:
    timestamp: np.int64
    id: np.int8
    alive: np.uint8
    position_x: np.float32
    position_y: np.float32
    position_z: np.float32
    orientation_x: np.float32
    orientation_y: np.float32
    orientation_z: np.float32
    velocity_x: np.float32
    velocity_y: np.float32
    velocity_z: np.float32
    angular_velocity_x: np.float32
    angular_velocity_y: np.float32
    angular_velocity_z: np.float32
    acceleration_x: np.float32
    acceleration_y: np.float32
    acceleration_z: np.float32

    def get_conversion_str(self) -> str:
        return "<q3f3f3f3f3f3f"

    def __len__(self) -> int:
        return self.size()

    @classmethod
    def size(cls) -> int:
        return 8 + 1 + 1 + 3 * 4 + 3 * 4 + 3 * 4 + 3 * 4 + 3 * 4

    @classmethod
    def from_buffer(cls, buffer: bytes) -> DroneData:
        if len(buffer) < cls.size():
            raise ValueError(
                f"Buffer size {len(buffer)} is smaller than expected size {cls.size()}."
            )
        return DroneData(
            timestamp=np.frombuffer(buffer[0:8], dtype=np.int64)[0],
            id=np.frombuffer(buffer[8:9], dtype=np.int8)[0],
            alive=np.frombuffer(buffer[9:10], dtype=np.uint8)[0],
            position_x=np.frombuffer(buffer[10:14], dtype=np.float32)[0],
            position_y=np.frombuffer(buffer[14:18], dtype=np.float32)[0],
            position_z=np.frombuffer(buffer[18:22], dtype=np.float32)[0],
            orientation_x=np.frombuffer(buffer[22:26], dtype=np.float32)[0],
            orientation_y=np.frombuffer(buffer[26:30], dtype=np.float32)[0],
            orientation_z=np.frombuffer(buffer[30:34], dtype=np.float32)[0],
            velocity_x=np.frombuffer(buffer[34:38], dtype=np.float32)[0],
            velocity_y=np.frombuffer(buffer[38:42], dtype=np.float32)[0],
            velocity_z=np.frombuffer(buffer[42:46], dtype=np.float32)[0],
            angular_velocity_x=np.frombuffer(buffer[46:50], dtype=np.float32)[0],
            angular_velocity_y=np.frombuffer(buffer[50:54], dtype=np.float32)[0],
            angular_velocity_z=np.frombuffer(buffer[54:58], dtype=np.float32)[0],
            acceleration_x=np.frombuffer(buffer[58:62], dtype=np.float32)[0],
            acceleration_y=np.frombuffer(buffer[62:66], dtype=np.float32)[0],
            acceleration_z=np.frombuffer(buffer[66:70], dtype=np.float32)[0],
        )

    @classmethod
    def from_dict(cls, data: dict) -> DroneData:
        try:
            return DroneData(
                timestamp=np.int64(data["timestamp"]),
                id=np.int8(data["id"]),
                alive=np.uint8(data["alive"]),
                position_x=np.float32(data["position"][0]),
                position_y=np.float32(data["position"][1]),
                position_z=np.float32(data["position"][2]),
                orientation_x=np.float32(data["orientation"][0]),
                orientation_y=np.float32(data["orientation"][1]),
                orientation_z=np.float32(data["orientation"][2]),
                velocity_x=np.float32(data["velocity"][0]),
                velocity_y=np.float32(data["velocity"][1]),
                velocity_z=np.float32(data["velocity"][2]),
                angular_velocity_x=np.float32(data["angular_velocity"][0]),
                angular_velocity_y=np.float32(data["angular_velocity"][1]),
                angular_velocity_z=np.float32(data["angular_velocity"][2]),
                acceleration_x=np.float32(data["acceleration"][0]),
                acceleration_y=np.float32(data["acceleration"][1]),
                acceleration_z=np.float32(data["acceleration"][2]),
            )
        except KeyError as e:
            raise ValueError(f"Missing key in data dictionary: {e}") from e

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DroneData):
            return NotImplemented
        return (
            self.timestamp == other.timestamp
            and self.id == other.id
            and self.alive == other.alive
        )


@dataclass
class UserInputData:
    timestamp: np.int64
    altitude_rate: np.float32
    yaw_rate: np.float32
    pitch_rate: np.float32
    roll_rate: np.float32
    swarm_spread: np.float32
    max_pitch: np.float32
    max_roll: np.float32
    max_yaw_rate: np.float32
    max_speed: np.float32
    max_altitude_rate: np.float32
    max_alpha: np.float32
    cwl_total_steps: np.int32
    cwl_current_step: np.int32

    def get_conversion_str(self) -> str:
        return "<q11f"

    def __len__(self) -> int:
        return self.size()

    @classmethod
    def size(cls) -> int:
        return 8 + 11 * 4 + 2 * 4

    @classmethod
    def from_buffer(cls, buffer: bytes) -> UserInputData:
        if len(buffer) < cls.size():
            raise ValueError(
                f"Buffer size {len(buffer)} is smaller than expected size {cls.size()}."
            )
        return UserInputData(
            timestamp=np.frombuffer(buffer[0:8], dtype=np.int64)[0],
            altitude_rate=np.frombuffer(buffer[8:12], dtype=np.float32)[0],
            yaw_rate=np.frombuffer(buffer[12:16], dtype=np.float32)[0],
            pitch_rate=np.frombuffer(buffer[16:20], dtype=np.float32)[0],
            roll_rate=np.frombuffer(buffer[20:24], dtype=np.float32)[0],
            swarm_spread=np.frombuffer(buffer[24:28], dtype=np.float32)[0],
            max_pitch=np.frombuffer(buffer[28:32], dtype=np.float32)[0],
            max_roll=np.frombuffer(buffer[32:36], dtype=np.float32)[0],
            max_yaw_rate=np.frombuffer(buffer[36:40], dtype=np.float32)[0],
            max_speed=np.frombuffer(buffer[40:44], dtype=np.float32)[0],
            max_altitude_rate=np.frombuffer(buffer[44:48], dtype=np.float32)[0],
            max_alpha=np.frombuffer(buffer[48:52], dtype=np.float32)[0],
            cwl_total_steps=np.frombuffer(buffer[52:56], dtype=np.int32)[0],
            cwl_current_step=np.frombuffer(buffer[56:60], dtype=np.int32)[0],
        )

    @classmethod
    def from_dict(cls, data: dict) -> UserInputData:
        try:
            return UserInputData(
                timestamp=np.int64(data["timestamp"]),
                altitude_rate=np.float32(data["altitude_rate"]),
                yaw_rate=np.float32(data["yaw_rate"]),
                pitch_rate=np.float32(data["pitch_rate"]),
                roll_rate=np.float32(data["roll_rate"]),
                swarm_spread=np.float32(data["swarm_spread"]),
                max_pitch=np.float32(data["max_pitch"]),
                max_roll=np.float32(data["max_roll"]),
                max_yaw_rate=np.float32(data["max_yaw_rate"]),
                max_speed=np.float32(data["max_speed"]),
                max_altitude_rate=np.float32(data["max_altitude_rate"]),
                max_alpha=np.float32(data["max_alpha"]),
                cwl_total_steps=np.int32(data["cwl_total_steps"]),
                cwl_current_step=np.int32(data["cwl_current_step"]),
            )
        except KeyError as e:
            raise ValueError(f"Missing key in data dictionary: {e}") from e


@dataclass
class GazeData:
    timestamp: np.int64
    left_gaze_point_x: np.float32
    left_gaze_point_y: np.float32
    left_gaze_point_z: np.float32
    right_gaze_point_x: np.float32
    right_gaze_point_y: np.float32
    right_gaze_point_z: np.float32
    left_point_screen_x: np.float32
    left_point_screen_y: np.float32
    right_point_screen_x: np.float32
    right_point_screen_y: np.float32
    left_validity: np.int8
    right_validity: np.int8
    left_pupil_diameter: np.float32
    right_pupil_diameter: np.float32
    left_openness_validity: np.int8 = np.int8(0)
    right_openness_validity: np.int8 = np.int8(0)
    left_openness: np.float32 = np.float32(0.0)
    right_openness: np.float32 = np.float32(0.0)

    def get_conversion_str(self) -> str:
        return "<q10f2B2f2B2f"

    def __len__(self) -> int:
        return self.size()

    @classmethod
    def size(cls) -> int:
        return 8 + 3 * 4 + 3 * 4 + 2 * 4 + 2 * 4 + 2 * 1 + 2 * 4 + 2 * 1 + 2 * 4

    @classmethod
    def from_buffer(cls, buffer: bytes) -> GazeData:
        return GazeData(
            timestamp=np.frombuffer(buffer[0:8], dtype=np.int64)[0],
            left_gaze_point_x=np.frombuffer(buffer[8:12], dtype=np.float32)[0],
            left_gaze_point_y=np.frombuffer(buffer[12:16], dtype=np.float32)[0],
            left_gaze_point_z=np.frombuffer(buffer[16:20], dtype=np.float32)[0],
            right_gaze_point_x=np.frombuffer(buffer[20:24], dtype=np.float32)[0],
            right_gaze_point_y=np.frombuffer(buffer[24:28], dtype=np.float32)[0],
            right_gaze_point_z=np.frombuffer(buffer[28:32], dtype=np.float32)[0],
            left_point_screen_x=np.frombuffer(buffer[32:36], dtype=np.float32)[0],
            left_point_screen_y=np.frombuffer(buffer[36:40], dtype=np.float32)[0],
            right_point_screen_x=np.frombuffer(buffer[40:44], dtype=np.float32)[0],
            right_point_screen_y=np.frombuffer(buffer[44:48], dtype=np.float32)[0],
            left_validity=np.frombuffer(buffer[48:49], dtype=np.int8)[0],
            right_validity=np.frombuffer(buffer[49:50], dtype=np.int8)[0],
            left_pupil_diameter=np.frombuffer(buffer[50:54], dtype=np.float32)[0],
            right_pupil_diameter=np.frombuffer(buffer[54:58], dtype=np.float32)[0],
            left_openness_validity=np.frombuffer(buffer[58:59], dtype=np.int8)[0],
            right_openness_validity=np.frombuffer(buffer[59:60], dtype=np.int8)[0],
            left_openness=np.frombuffer(buffer[60:64], dtype=np.float32)[0],
            right_openness=np.frombuffer(buffer[64:68], dtype=np.float32)[0],
        )

    @classmethod
    def from_dict(cls, data: dict) -> GazeData:
        try:
            return GazeData(
                timestamp=np.int64(data["system_time_stamp"]),
                left_gaze_point_x=np.float32(
                    data["left_gaze_origin_in_user_coordinate_system"][0]
                ),
                left_gaze_point_y=np.float32(
                    data["left_gaze_origin_in_user_coordinate_system"][1]
                ),
                left_gaze_point_z=np.float32(
                    data["left_gaze_origin_in_user_coordinate_system"][2]
                ),
                right_gaze_point_x=np.float32(
                    data["right_gaze_origin_in_user_coordinate_system"][0]
                ),
                right_gaze_point_y=np.float32(
                    data["right_gaze_origin_in_user_coordinate_system"][1]
                ),
                right_gaze_point_z=np.float32(
                    data["right_gaze_origin_in_user_coordinate_system"][2]
                ),
                left_point_screen_x=np.float32(
                    data["left_gaze_point_on_display_area"][0]
                ),
                left_point_screen_y=np.float32(
                    data["left_gaze_point_on_display_area"][1]
                ),
                right_point_screen_x=np.float32(
                    data["right_gaze_point_on_display_area"][0]
                ),
                right_point_screen_y=np.float32(
                    data["right_gaze_point_on_display_area"][1]
                ),
                left_validity=np.int8(data["left_pupil_validity"]),
                right_validity=np.int8(data["right_pupil_validity"]),
                left_pupil_diameter=np.float32(data["left_pupil_diameter"]),
                right_pupil_diameter=np.float32(data["right_pupil_diameter"]),
                left_openness_validity=np.int8(
                    data.get("left_eye_openness_validity", 0)
                ),
                right_openness_validity=np.int8(
                    data.get("right_eye_openness_validity", 0)
                ),
                left_openness=np.float32(data.get("left_eye_openness", 0.0)),
                right_openness=np.float32(data.get("right_eye_openness", 0.0)),
            )
        except KeyError as e:
            raise ValueError(f"Missing key in data dictionary: {e}") from e


class ExperimentState(Enum):
    Idle = 0
    IdleSilent = 1
    Wait = 2
    WaitForUser = 3
    Welcome = 4
    RcControls = 5
    Calibration = 6
    FlyingInstructions = 7
    FlyingPractice = 8
    NBackInstructions = 9
    NBackPractice = 10
    ExperimentBegin = 11
    Task = 12
    Countdown = 13
    Trial = 14
    Finished = 15
    ReadyScreen = 16
    RaceInstructions = 17


@dataclass
class InferenceRecord:
    timestamp: int
    prob_low: float
    prob_medium: float
    prob_high: float
    raw_state: int
    filtered_state: int
    nback_level: int


@dataclass
class ExperimentStatus:
    previous_state: ExperimentState
    current_state: ExperimentState
    next_state: ExperimentState
    current_task: int
    total_tasks: int
    current_trial: int
    total_trials: int
    nback_levels_order: list[int]
    current_nback_level: int
    state_enter_timestamp: np.int64

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentStatus:
        try:
            return ExperimentStatus(
                previous_state=ExperimentState[data["previousState"]],
                current_state=ExperimentState[data["state"]],
                next_state=ExperimentState[data["nextState"]],
                current_task=data["currentTask"],
                total_tasks=data["totalTaskNumber"],
                current_trial=data["currentTrial"],
                total_trials=data["totalTrialNumber"],
                nback_levels_order=data.get(
                    "nBackLevelsOrder", []
                ),  # Optional field, default to empty list if not present
                state_enter_timestamp=np.int64(data["stateEnterTimestamp"]),
                current_nback_level=data.get(
                    "currentNBackLevel", -1
                ),  # Optional field, default to -1 if not present
            )
        except KeyError as e:
            raise ValueError(f"Missing key in data dictionary: {e}") from e
        except ValueError as e:
            raise ValueError(f"Invalid value in data dictionary: {e}") from e


@dataclass
class GateLayoutEntry:
    """Static gate layout written once when course is generated.

    SM block: ExperimentUnityGateLayout
    Binary layout (pack=1):
        id       : 1 byte  (uint8)
        center_x : 4 bytes (float32)
        center_y : 4 bytes (float32)
        center_z : 4 bytes (float32)
        width    : 4 bytes (float32)
        height   : 4 bytes (float32)
        total    : 21 bytes
    """

    id: np.uint8
    is_hard: np.uint8
    center_x: np.float32
    center_y: np.float32
    center_z: np.float32
    width: np.float32
    height: np.float32

    @classmethod
    def size(cls) -> int:
        return 1 + 1 + 4 + 4 + 4 + 4 + 4  # 21 bytes

    @classmethod
    def from_buffer(cls, buffer: bytes) -> "GateLayoutEntry":
        if len(buffer) < cls.size():
            raise ValueError(
                f"Buffer size {len(buffer)} is smaller than expected {cls.size()}."
            )
        offset = 0
        id_ = np.frombuffer(buffer[offset : offset + 1], dtype=np.uint8)[0]
        offset += 1
        is_hard = np.frombuffer(buffer[offset : offset + 1], dtype=np.uint8)[0]
        offset += 1
        center_x = np.frombuffer(buffer[offset : offset + 4], dtype=np.float32)[0]
        offset += 4
        center_y = np.frombuffer(buffer[offset : offset + 4], dtype=np.float32)[0]
        offset += 4
        center_z = np.frombuffer(buffer[offset : offset + 4], dtype=np.float32)[0]
        offset += 4
        width = np.frombuffer(buffer[offset : offset + 4], dtype=np.float32)[0]
        offset += 4
        height = np.frombuffer(buffer[offset : offset + 4], dtype=np.float32)[0]
        return cls(
            id=id_,
            is_hard=is_hard,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            width=width,
            height=height,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "GateLayoutEntry":
        return cls(
            id=np.uint8(data["id"]),
            is_hard=np.uint8(
                data.get("is_hard", 0)
            ),  # Optional, default to 0 if not present
            center_x=np.float32(data["center"][0]),
            center_y=np.float32(data["center"][1]),
            center_z=np.float32(data["center"][2]),
            width=np.float32(data["width"]),
            height=np.float32(data["height"]),
        )


@dataclass
class GateStatusEntry:
    """Real-time gate status per gate, updated as drones pass through gates.

    SM block: ExperimentUnityGateStatus
    Binary layout (pack=1):
        id                   : 1 byte  (uint8)
        pass_count           : 1 byte  (uint8)   — number of drones that passed
        gate_state           : 1 byte  (uint8)   — 0=Idle 1=Next 2=PartialComplete 3=Completed
        first_pass_timestamp : 8 bytes (int64)   — Unix ms, 0 if not yet passed
        total                : 11 bytes
    """

    id: np.uint8
    pass_count: np.uint8
    gate_state: np.uint8
    first_pass_timestamp: np.int64

    @classmethod
    def size(cls) -> int:
        return 1 + 1 + 1 + 8  # 11 bytes

    @classmethod
    def from_buffer(cls, buffer: bytes) -> "GateStatusEntry":
        if len(buffer) < cls.size():
            raise ValueError(
                f"Buffer size {len(buffer)} is smaller than expected {cls.size()}."
            )
        offset = 0
        id_ = np.frombuffer(buffer[offset : offset + 1], dtype=np.uint8)[0]
        offset += 1
        pass_count = np.frombuffer(buffer[offset : offset + 1], dtype=np.uint8)[0]
        offset += 1
        gate_state = np.frombuffer(buffer[offset : offset + 1], dtype=np.uint8)[0]
        offset += 1
        first_pass_timestamp = np.frombuffer(
            buffer[offset : offset + 8], dtype=np.int64
        )[0]
        return cls(
            id=id_,
            pass_count=pass_count,
            gate_state=gate_state,
            first_pass_timestamp=first_pass_timestamp,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "GateStatusEntry":
        return cls(
            id=np.uint8(data["id"]),
            pass_count=np.uint8(data["pass_count"]),
            gate_state=np.uint8(data["gate_state"]),
            first_pass_timestamp=np.int64(data["first_pass_timestamp"]),
        )
