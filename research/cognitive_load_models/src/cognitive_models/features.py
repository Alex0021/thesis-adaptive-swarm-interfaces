from collections import defaultdict
from typing import Any

import pandas as pd

from .gaze_utils import calculate_fixations_saccades
from .pupil_utils import lhipa


def extract_window_features(
    window_df: pd.DataFrame,
    blink_df: pd.DataFrame,
    ivt_threshold: int,
    min_fixation_duration: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Extracts features from the eye-tracking window data, including fixations, saccades, and blinks.

    :param window_df:With columns 'timestamp_sec', 'confidence', 'blink', 'gaze_angle_delta_deg', and 'gaze_angular_velocity'.
    :param blink_df: DataFrame containing blink information with columns 'start_timestamp', 'stop_timestamp'.
    :param ivt_threshold: The velocity threshold (in deg/s) for identifying saccades.
    :param min_fixation_duration: The minimum duration (in milliseconds) for a fixation to be considered valid.
    :return features: A dictionary containing extracted features.
    """
    # 1- Extract fixations and saccades
    _, fixations, saccades = calculate_fixations_saccades(
        window_df, ivt_threshold, min_fixation_duration, verbose=verbose
    )

    features = defaultdict(lambda: 0)
    # 2- Fixations: count, duration mean/max/min/std
    features["fixations_count"] = len(fixations)
    if not fixations.empty:
        features["fixations_duration_mean"] = fixations["duration_ms"].mean()
        features["fixations_duration_max"] = fixations["duration_ms"].max()
        features["fixations_duration_min"] = fixations["duration_ms"].min()
        features["fixations_duration_std"] = (
            fixations["duration_ms"].std() if len(fixations) > 1 else 0
        )

    # 3- Saccades: count, peak_velocity, amplitude mean/max/min/std, duration mean/max/min/std
    features["saccades_count"] = len(saccades)
    if not saccades.empty:
        features["saccades_peak_velocity_mean"] = saccades["peak_velocity"].mean()
        features["saccades_amplitude_mean"] = saccades["amplitude_deg"].mean()
        features["saccades_amplitude_max"] = saccades["amplitude_deg"].max()
        features["saccades_amplitude_min"] = saccades["amplitude_deg"].min()
        features["saccades_amplitude_std"] = (
            saccades["amplitude_deg"].std() if len(saccades) > 1 else 0
        )
        features["saccades_duration_mean"] = saccades["duration_ms"].mean()
        features["saccades_duration_max"] = saccades["duration_ms"].max()
        features["saccades_duration_min"] = saccades["duration_ms"].min()
        features["saccades_duration_std"] = (
            saccades["duration_ms"].std() if len(saccades) > 1 else 0
        )

    # 4- Blinks: count, duration mean
    if blink_df is None:
        features["blinks_count"] = 0
        features["blinks_duration_mean"] = 0
    else:
        features["blinks_count"] = len(blink_df)
        features["blinks_duration_mean"] = 0
        if not blink_df.empty:
            features["blinks_duration_mean"] = (
                blink_df["stop_timestamp"] - blink_df["start_timestamp"]
            ).mean()

    # 5- Pupil related features
    features["pupil_lhipa"] = lhipa(window_df)

    return features
