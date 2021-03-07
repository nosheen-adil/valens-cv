from valens import constants
from valens import structures as core
import valens.structures.feedback
from valens.structures.pose import Keypoints

import numpy as np
import cv2
import h5py

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

    result = core.feedback.to_json(pose, corrected_pose=expected_pose, keypoints=keypoints)
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

def test_feedback_topology():
    keypoints = [Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK]
    topology = [[0, 1], [1, 2], [2, 3]] # [ankle -> knee, knee -> hip, hip -> neck]
    feedback_topology = core.feedback.topology(topology, keypoints)
    assert feedback_topology == [['right_ankle', 'right_knee'], ['right_knee', 'right_hip'], ['right_hip', 'neck']]

def test_feedback_draw_on_image_correct():
    keypoints = [Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK]
    topology = [[0, 1], [1, 2], [2, 3]] # [ankle -> knee, knee -> hip, hip -> neck]
    feedback_topology = core.feedback.topology(topology, keypoints)

    pose = np.array([
        [0.5, 0.7],
        [0.5, 0.5],
        [0.5, 0.35],
        [0.5, 0.1]
    ])
    feedback = {
        'pose' : {
            'right_ankle' : {
                'x' : pose[0, 0],
                'y' : pose[0, 1]
            },
            'right_knee' : {
                'x' : pose[1, 0],
                'y' : pose[1, 1]
            },
            'right_hip' : {
                'x' : pose[2, 0],
                'y' : pose[2, 1]
            },
            'neck' : {
                'x' : pose[3, 0],
                'y' : pose[3, 1]
            }
        },
        'feedback' : {
            'right_ankle' : {
                'correct' : True
            },
            'right_knee' : {
                'correct' : True
            },
            'right_hip' : {
                'correct' : True
            },
            'neck' : {
                'correct' : True
            }
        }
    }

    expected_frame = np.zeros((constants.POSE_MODEL_WIDTH, constants.POSE_MODEL_HEIGHT, 3), dtype=np.uint8)
    core.pose.draw_on_image(pose, expected_frame, topology)

    actual_frame = np.zeros((constants.POSE_MODEL_WIDTH, constants.POSE_MODEL_HEIGHT, 3), dtype=np.uint8)
    core.feedback.draw_on_image(feedback, actual_frame, feedback_topology)

    np.testing.assert_equal(actual_frame, expected_frame)

def test_feedback_draw_on_image_incorrect():
    keypoints = [Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK]
    topology = [[0, 1], [1, 2], [2, 3]] # [ankle -> knee, knee -> hip, hip -> neck]
    feedback_topology = core.feedback.topology(topology, keypoints)

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

    feedback = {
        'pose' : {
            'right_ankle' : {
                'x' : pose[0, 0],
                'y' : pose[0, 1]
            },
            'right_knee' : {
                'x' : pose[1, 0],
                'y' : pose[1, 1]
            },
            'right_hip' : {
                'x' : pose[2, 0],
                'y' : pose[2, 1]
            },
            'neck' : {
                'x' : pose[3, 0],
                'y' : pose[3, 1]
            }
        },
        'feedback' : {
            'right_ankle' : {
                'correct' : True
            },
            'right_knee' : {
                'correct' : True
            },
            'right_hip' : {
                'correct' : False,
                'x' : expected_pose[2, 0],
                'y' : expected_pose[2, 1]
            },
            'neck' : {
                'correct' : False,
                'x' : expected_pose[3, 0],
                'y' : expected_pose[3, 1]
            }
        }
    }

    actual_frame = np.zeros((constants.POSE_MODEL_WIDTH, constants.POSE_MODEL_HEIGHT, 3), dtype=np.uint8)
    core.feedback.draw_on_image(feedback, actual_frame, feedback_topology)

    expected_filename = constants.TEST_DATA_DIR + '/feedback_draw_on_image_incorrect.h5'
    with h5py.File(expected_filename, 'r') as data:
        expected_frame = data['image'][:]
    
    np.testing.assert_equal(actual_frame, expected_frame)
