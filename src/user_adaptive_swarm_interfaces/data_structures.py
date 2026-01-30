from dataclasses import dataclass
import numpy as np

@dataclass 
class Metadata:
    stream_ready: np.uint8 = 0
    calibration_ok: np.uint8 = 0
    active_data_cnt: np.uint8 = 0

    def get_conversion_str(self) -> str:
        return 'BBB'

    def __len__(self) -> int:
        return self.size()

    @classmethod
    def size(cls) -> int:
        return 1 + 1 + 1
    
    @classmethod
    def from_buffer(cls, buffer: bytes) -> "Metadata":
        return Metadata(
            stream_ready = np.frombuffer(buffer[0:1], dtype=np.uint8)[0],
            calibration_ok = np.frombuffer(buffer[1:2], dtype=np.uint8)[0],
            active_data_cnt = np.frombuffer(buffer[2:3], dtype=np.uint8)[0],
        )

@dataclass
class GazeData:
    timestamp: np.int64
    left_gaze_point: np.ndarray[np.float32]
    right_gaze_point: np.ndarray[np.float32]
    left_point_screen: np.ndarray[np.float32]
    right_point_screen: np.ndarray[np.float32]
    left_validity: np.int8
    right_validity: np.int8
    left_pupil_diameter: np.float32
    right_pupil_diameter: np.float32

    def get_conversion_str(self) -> str:
        return '<q10f2B2f'

    def __len__(self) -> int:
        return self.size()

    @classmethod
    def size(cls) -> int:
        return 8 + 3*4 + 3*4 + 2*4 + 2*4 + 1 + 1 + 4 + 4

    @classmethod
    def from_buffer(cls, buffer: bytes) -> "GazeData":
        return GazeData(
            timestamp = np.frombuffer(buffer[0:8], dtype=np.int64)[0],
            left_gaze_point = np.frombuffer(buffer[8:20], dtype=np.float32),
            right_gaze_point = np.frombuffer(buffer[20:32], dtype=np.float32),
            left_point_screen = np.frombuffer(buffer[32:40], dtype=np.float32),
            right_point_screen = np.frombuffer(buffer[40:48], dtype=np.float32),
            left_validity = np.frombuffer(buffer[48:49], dtype=np.int8)[0],
            right_validity = np.frombuffer(buffer[49:50], dtype=np.int8)[0],
            left_pupil_diameter = np.frombuffer(buffer[50:54], dtype=np.float32)[0],
            right_pupil_diameter = np.frombuffer(buffer[54:58], dtype=np.float32)[0],
        )