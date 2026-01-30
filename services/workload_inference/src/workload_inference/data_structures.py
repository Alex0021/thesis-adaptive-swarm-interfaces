from dataclasses import dataclass
import numpy as np

@dataclass 
class Metadata:
    stream_ready: np.uint8
    calibration_ok: np.uint8
    active_data_cnt: np.uint8

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
            left_gaze_point_x = np.frombuffer(buffer[8:12], dtype=np.float32)[0],
            left_gaze_point_y = np.frombuffer(buffer[12:16], dtype=np.float32)[0],
            left_gaze_point_z = np.frombuffer(buffer[16:20], dtype=np.float32)[0],
            right_gaze_point_x = np.frombuffer(buffer[20:24], dtype=np.float32)[0],
            right_gaze_point_y = np.frombuffer(buffer[24:28], dtype=np.float32)[0],
            right_gaze_point_z = np.frombuffer(buffer[28:32], dtype=np.float32)[0],
            left_point_screen_x = np.frombuffer(buffer[32:36], dtype=np.float32)[0],
            left_point_screen_y = np.frombuffer(buffer[36:40], dtype=np.float32)[0],
            right_point_screen_x = np.frombuffer(buffer[40:44], dtype=np.float32)[0],
            right_point_screen_y = np.frombuffer(buffer[44:48], dtype=np.float32)[0],
            left_validity = np.frombuffer(buffer[48:49], dtype=np.int8)[0],
            right_validity = np.frombuffer(buffer[49:50], dtype=np.int8)[0],
            left_pupil_diameter = np.frombuffer(buffer[50:54], dtype=np.float32)[0],
            right_pupil_diameter = np.frombuffer(buffer[54:58], dtype=np.float32)[0],
        )