"""Constants for the G1 parkour sim2sim runner."""

from __future__ import annotations

TRAINING_JOINT_ORDER = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

ACTION_SCALE_BY_NAME = {
    "left_shoulder_pitch_joint": 0.43857731392336724,
    "right_shoulder_pitch_joint": 0.43857731392336724,
    "waist_pitch_joint": 0.43857731392336724,
    "left_shoulder_roll_joint": 0.43857731392336724,
    "right_shoulder_roll_joint": 0.43857731392336724,
    "waist_roll_joint": 0.43857731392336724,
    "left_shoulder_yaw_joint": 0.43857731392336724,
    "right_shoulder_yaw_joint": 0.43857731392336724,
    "waist_yaw_joint": 0.5475464652142303,
    "left_elbow_joint": 0.43857731392336724,
    "right_elbow_joint": 0.43857731392336724,
    "left_hip_pitch_joint": 0.5475464652142303,
    "right_hip_pitch_joint": 0.5475464652142303,
    "left_wrist_roll_joint": 0.43857731392336724,
    "right_wrist_roll_joint": 0.43857731392336724,
    "left_hip_roll_joint": 0.3506614663788243,
    "right_hip_roll_joint": 0.3506614663788243,
    "left_wrist_pitch_joint": 0.07450087032950714,
    "right_wrist_pitch_joint": 0.07450087032950714,
    "left_hip_yaw_joint": 0.5475464652142303,
    "right_hip_yaw_joint": 0.5475464652142303,
    "left_wrist_yaw_joint": 0.07450087032950714,
    "right_wrist_yaw_joint": 0.07450087032950714,
    "left_knee_joint": 0.3506614663788243,
    "right_knee_joint": 0.3506614663788243,
    "left_ankle_pitch_joint": 0.43857731392336724,
    "right_ankle_pitch_joint": 0.43857731392336724,
    "left_ankle_roll_joint": 0.43857731392336724,
    "right_ankle_roll_joint": 0.43857731392336724,
}

KP_BY_NAME = {
    ".*_hip_pitch_joint": 40.17923847137318,
    ".*_hip_roll_joint": 99.09842777666113,
    ".*_hip_yaw_joint": 40.17923847137318,
    ".*_knee_joint": 99.09842777666113,
    ".*_ankle_pitch_joint": 28.50124619574858,
    ".*_ankle_roll_joint": 28.50124619574858,
    "waist_roll_joint": 28.50124619574858,
    "waist_pitch_joint": 28.50124619574858,
    "waist_yaw_joint": 40.17923847137318,
    ".*_shoulder_pitch_joint": 14.25062309787429,
    ".*_shoulder_roll_joint": 14.25062309787429,
    ".*_shoulder_yaw_joint": 14.25062309787429,
    ".*_elbow_joint": 14.25062309787429,
    ".*_wrist_roll_joint": 14.25062309787429,
    ".*_wrist_pitch_joint": 16.77832748089279,
    ".*_wrist_yaw_joint": 16.77832748089279,
}

KD_BY_NAME = {
    ".*_hip_pitch_joint": 2.5578897650279457,
    ".*_hip_roll_joint": 6.3088018534966395,
    ".*_hip_yaw_joint": 2.5578897650279457,
    ".*_knee_joint": 6.3088018534966395,
    ".*_ankle_pitch_joint": 1.814445686584846,
    ".*_ankle_roll_joint": 1.814445686584846,
    "waist_roll_joint": 1.814445686584846,
    "waist_pitch_joint": 1.814445686584846,
    "waist_yaw_joint": 2.5578897650279457,
    ".*_shoulder_pitch_joint": 0.907222843292423,
    ".*_shoulder_roll_joint": 0.907222843292423,
    ".*_shoulder_yaw_joint": 0.907222843292423,
    ".*_elbow_joint": 0.907222843292423,
    ".*_wrist_roll_joint": 0.907222843292423,
    ".*_wrist_pitch_joint": 1.06814150219,
    ".*_wrist_yaw_joint": 1.06814150219,
}

EFFORT_LIMIT_BY_NAME = {
    ".*_hip_yaw_joint": 88.0,
    ".*_hip_roll_joint": 139.0,
    ".*_hip_pitch_joint": 88.0,
    ".*_knee_joint": 139.0,
    ".*_ankle_pitch_joint": 50.0,
    ".*_ankle_roll_joint": 50.0,
    "waist_roll_joint": 50.0,
    "waist_pitch_joint": 50.0,
    "waist_yaw_joint": 88.0,
    ".*_shoulder_pitch_joint": 25.0,
    ".*_shoulder_roll_joint": 25.0,
    ".*_shoulder_yaw_joint": 25.0,
    ".*_elbow_joint": 25.0,
    ".*_wrist_roll_joint": 25.0,
    ".*_wrist_pitch_joint": 5.0,
    ".*_wrist_yaw_joint": 5.0,
}


def _lookup(pattern_map: dict[str, float], joint_name: str) -> float:
    if joint_name in pattern_map:
        return pattern_map[joint_name]
    for pattern, value in pattern_map.items():
        if pattern.startswith(".*") and joint_name.endswith(pattern[2:]):
            return value
    raise KeyError(f"No value configured for joint {joint_name!r}")


def values_in_training_order(pattern_map: dict[str, float]) -> list[float]:
    return [_lookup(pattern_map, name) for name in TRAINING_JOINT_ORDER]
