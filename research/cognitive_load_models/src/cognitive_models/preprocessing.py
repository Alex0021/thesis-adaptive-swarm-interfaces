import glob
from pathlib import Path

import pandas as pd

from cognitive_models.pupil_utils import detect_outliers

from .gaze_utils import (
    calculate_angular_velocity,
    calculate_gaze_angular_delta,
    detect_gaps_and_blinks,
)
from .interpolate import (
    interpolate_blinks,
    interpolate_gaze_angle,
    interpolate_missing_gaze,
    interpolate_pupil_data,
    merge_colet_eye_data,
)

# COLET dataset file patterns
PARTICIPANT_FODLER_PATTERN = "participant_{:02d}"
TASK_FOLDER_PATTERN = "Task_{:01d}"
GAZE_FILE_PATTERN = "*gaze.csv"
PUPIL_FILE_PATTERN = "*pupil.csv"
BLINK_FILE_PATTERN = "*blinks.csv"
ANNOTATION_FILENAME = "annotations.csv"

CONFIDENCE_THRESHOLD = 0.95
DURATION_THRESHOLD = 75 / 1000  # 75 ms in seconds
INTERPOLATION_THRESHOLD = 300 / 1000


def load_colet_data(dataset_dir: str, subject_ids: list[int], task_ids: list[int]):
    """
    Loads the Colet dataset for the specified subjects and task IDs
    Args:
        dataset_dir (str): The base directory where the dataset is stored.
        subject_ids (list[int]): A list of subject IDs to load.
        task_ids (list[int]): A list of task IDs to load.
    Returns:
        dataframe: A pandas DataFrame containing the loaded data.
    """
    if not Path(dataset_dir).exists():
        raise FileNotFoundError(
            f"The specified dataset directory '{dataset_dir}' does not exist."
        )

    all_eye_df = pd.DataFrame()
    for subject_id in subject_ids:
        subject_path = Path(dataset_dir) / PARTICIPANT_FODLER_PATTERN.format(subject_id)
        # In every subject folder, there is a "annotations.csv" file containing
        # the NASA TLX scores for each task. Add it also to the dataframe.
        # Only keep mean score for each task
        annotation_file = subject_path / ANNOTATION_FILENAME
        annotation_df = None
        if annotation_file.exists():
            annotation_df = pd.read_csv(annotation_file, header=0)
            annotation_df["mean_score"] = annotation_df.mean(axis=1, numeric_only=True)

        for task_id in task_ids:
            task_path = subject_path / TASK_FOLDER_PATTERN.format(task_id)
            gaze_file = glob.glob(str(subject_path / task_path / GAZE_FILE_PATTERN))
            pupil_file = glob.glob(str(subject_path / task_path / PUPIL_FILE_PATTERN))
            blink_file = glob.glob(str(subject_path / task_path / BLINK_FILE_PATTERN))

            if not gaze_file or not pupil_file or not blink_file:
                print(
                    f"Warning: Missing data for participant {subject_id}, task {task_id}. Skipping."
                )
                continue

            # Load the data files into DataFrames
            gaze_df = pd.read_csv(gaze_file[0])
            pupil_df = pd.read_csv(pupil_file[0])

            # Merge gaze and pupil data, keep blink data separate for now
            merged_df = merge_colet_eye_data(gaze_df, pupil_df, f_des=60)

            # Add subject and task identifiers to the DataFrame
            merged_df["subject_id"] = subject_id
            merged_df["task_id"] = task_id

            # Add CL estimation from NASA RTX annotations
            if annotation_df is not None:
                mean_score = annotation_df.iloc[task_id - 1]["mean_score"]
                merged_df["cl_class"] = (
                    "high"
                    if mean_score > 49
                    else ("low" if mean_score < 30 else "medium")
                )

            # Append the merged DataFrame to the overall DataFrame
            all_eye_df = pd.concat([all_eye_df, merged_df], ignore_index=True)

    return all_eye_df


def preprocess_colet_data(
    eye_df: pd.DataFrame,
    min_num_samples: int = 5,
    margins: int = 50 / 1000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the Colet dataset by interpolating blinks and low condifence zones.

    Extract blinks & gaps --> calculate gaze angles --> interpolate blinks
      --> interpolate low confidence gaps --> calculate gaze angular velocity

    :param eye_df: DataFrame containing the raw eye-tracking data with columns 'timestamp_sec', 'confidence', and 'blink'.
    :param inter_window_N: The number of samples to consider for fitting interpolation function.
    :param min_num_samples: The minimum number of samples required on either side of a blink for interpolation.
    :param margins: The number of original samples to also inteprolate around the gaps / blinks (avoid edge effects).
    :param verbose: Whether to print progress information during preprocessing.
    :return inter_eye_df, custom_gaze_df, custom_pupil_df, gaps_to_fill_df: The preprocessed DataFrame with interpolated values and the custom blinks DataFrame.
    """
    eye_df = eye_df.copy()
    # Identify blinks and low confidence gaps
    gaps_to_fill_df = detect_gaps_and_blinks(eye_df)

    if verbose:
        print(
            f"Identified {len(gaps_to_fill_df[gaps_to_fill_df['is_blink']])} blinks and {len(gaps_to_fill_df)} low confidence gaps to fill."
        )

    # Add percentage of low confidence samples w.r.t total samples
    total_samples = len(eye_df)
    low_confidence_samples = (
        gaps_to_fill_df["stop_id"] - gaps_to_fill_df["start_id"] + 1
    ).sum()
    blink_samples = (
        gaps_to_fill_df[gaps_to_fill_df["is_blink"]]["stop_id"]
        - gaps_to_fill_df[gaps_to_fill_df["is_blink"]]["start_id"]
        + 1
    ).sum()
    eye_df["low_confidence_percentage"] = (
        low_confidence_samples / (total_samples - blink_samples) * 100
    )

    # Remove low confidence samples
    n_to_remove = eye_df[eye_df["confidence"] < CONFIDENCE_THRESHOLD].shape[0]
    eye_df = eye_df[eye_df["confidence"] >= CONFIDENCE_THRESHOLD]
    if verbose:
        print(f"Removed {n_to_remove} low confidence samples from the window.")

    # Remove outliers
    outliers_df = detect_outliers(eye_df, column="pupil_diameter_px", n_multiplier=10)
    eye_df = eye_df[~eye_df["timestamp_sec"].isin(outliers_df["timestamp_sec"])]
    if verbose:
        print(
            f"Removed {outliers_df.shape[0]} pupil diameter outliers from the window."
        )

    # Remove samples that are within the margins of detected blinks and gaps
    size_before = eye_df.shape[0]
    for _, row in gaps_to_fill_df[
        gaps_to_fill_df["duration_ms"] >= DURATION_THRESHOLD
    ].iterrows():
        idx_to_drop = eye_df[
            (eye_df["timestamp_sec"] >= row["start_timestamp"] - margins)
            & (eye_df["timestamp_sec"] <= row["stop_timestamp"] + margins)
        ].index
        eye_df.drop(idx_to_drop, inplace=True)
    size_after = eye_df.shape[0]
    if verbose:
        print(
            f"Removed {size_before - size_after} samples due to low confidence and proximity to detected blinks/gaps."
        )

    # Calculate gaze angles
    eye_df["gaze_angle_delta_deg"] = calculate_gaze_angular_delta(eye_df)

    # Interpolate data
    pupil_df_inter = interpolate_pupil_data(
        eye_df,
        gaps_to_fill_df,
        column="pupil_diameter_px",
        max_gap=INTERPOLATION_THRESHOLD,
    )
    gaze_df_inter = interpolate_gaze_angle(
        eye_df,
        gaps_to_fill_df,
        columns=["gaze_angle_delta_deg", "norm_pos_x", "norm_pos_y"],
        max_gap=INTERPOLATION_THRESHOLD,
    )

    # Finally, calculate gaze angular velocity
    gaze_df_inter["gaze_angular_velocity"] = calculate_angular_velocity(gaze_df_inter)

    return eye_df, gaze_df_inter, pupil_df_inter, gaps_to_fill_df
