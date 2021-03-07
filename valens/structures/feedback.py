from valens.structures.pose import Keypoints
import numpy as np

def to_json(user_pose, corrected_pose=None, keypoints=[]):
    if corrected_pose is not None:
        assert user_pose.shape == corrected_pose.shape

    if not keypoints:
        keypoints = [k for k in Keypoints]
    else:
        assert user_pose.shape[0] == len(keypoints)
    keypoint_names = [str(k) for k in keypoints]

    result = {
        'pose' : {},
        'feedback' : {}
    }

    for i, keypoint_name in enumerate(keypoint_names):
        result['pose'][keypoint_name] = {
            'x' : user_pose[i, 0],
            'y' : user_pose[i, 1]
        }

        if corrected_pose is None or np.allclose(user_pose[i, :], corrected_pose[i, :]):
            result['feedback'][keypoint_name] = {
                'correct' : True
            }
        else:
            result['feedback'][keypoint_name] = {
                'correct' : False,
                'x' : corrected_pose[i, 0],
                'y' : corrected_pose[i, 1]
            }

    return result
