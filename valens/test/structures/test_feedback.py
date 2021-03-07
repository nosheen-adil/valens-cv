from valens import constants
from valens import structures as core
import valens.structures.feedback
from valens.structures.pose import Keypoints

import numpy as np

def test_feedback_to_json_correct():
    keypoints = [Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK]
    pose = np.array([
        [0.5, 0.7],
        [0.5, 0.5],
        [0.5, 0.35],
        [0.5, 0.1]
    ])

    result = core.feedback.to_json(pose, keypoints=keypoints)
    for i, keypoint in enumerate([str(k) for k in keypoints]):
        assert result['pose'][keypoint]['x'] == pose[i, 0]
        assert result['pose'][keypoint]['y'] == pose[i, 1]
        assert result['feedback'][keypoint]['correct'] == True

def test_feedback_to_json_incorrect():
    keypoints = [Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK]
    pose = np.array([
        [0.5, 0.7],
        [0.5, 0.5],
        [0.5, 0.35],
        [0.5, 0.1]
    ])

    expected_pose = pose.copy()
    expected_pose[2, 0] -= abs(pose[2, 1] - pose[1, 1])
    expected_pose[2, 1] = pose[1, 1]
    expected_pose[3, 1] += abs(pose[2, 1] - pose[1, 1])
    expected_pose[3, 0] = expected_pose[2, 0]

    result = core.feedback.to_json(pose, expected_pose, keypoints)
    for i, keypoint in [[0, 'right_ankle'], [1, 'right_knee']]:
        assert result['pose'][keypoint]['x'] == pose[i, 0]
        assert result['pose'][keypoint]['y'] == pose[i, 1]
        assert result['feedback'][keypoint]['correct'] == True

    for i, keypoint in [[2, 'right_hip'], [3, 'neck']]:
        assert result['pose'][keypoint]['x'] == pose[i, 0]
        assert result['pose'][keypoint]['y'] == pose[i, 1]
        assert result['feedback'][keypoint]['correct'] == False
        assert result['feedback'][keypoint]['x'] == expected_pose[i, 0]
        assert result['feedback'][keypoint]['y'] == expected_pose[i, 1]
    