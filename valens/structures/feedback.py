from valens import constants
from valens.structures.pose import Keypoints
import numpy as np
import cv2

from enum import Enum

class KeypointResult(Enum):
    GOOD = 'good'
    BAD = 'bad'
    UNDEFINED = 'undefined'

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
        if corrected_pose is None:
            result['feedback'][keypoint_name] = {
                'correct' : None
            }
        elif np.allclose(user_pose[i, :], corrected_pose[i, :], rtol=1e-1):
            # print('close!')
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

def topology(topology, keypoints):
    result = []
    for link in topology:
        k1, k2 = link
        result.append([str(keypoints[k1]), str(keypoints[k2])])
    return result
    

def scale(feedback, width=constants.POSE_MODEL_WIDTH, height=constants.POSE_MODEL_HEIGHT):
    result = feedback.copy()
    for keypoint in result['pose'].keys():
        if result['pose'][keypoint]['x'] == np.nan:
            result['pose'][keypoint]['x'] = -1
        else:
            result['pose'][keypoint]['x'] = round(result['pose'][keypoint]['x'] * width)

        if result['pose'][keypoint]['y'] == np.nan:
            result['pose'][keypoint]['y'] = -1
        else:
            result['pose'][keypoint]['y'] = round(result['pose'][keypoint]['y'] * height)

    for keypoint in result['feedback'].keys():
        if result['feedback'][keypoint]['correct'] is False:
            result['feedback'][keypoint]['x'] = round(result['feedback'][keypoint]['x'] * width)
            result['feedback'][keypoint]['y'] = round(result['feedback'][keypoint]['y'] * height)

    return result

def draw_on_image(feedback, frame, topology, user_colors={'good' : (0, 153, 0), 'bad' : (0, 0, 204), 'undefined' : (255, 255, 255)}, feedback_color=(0, 153, 0)):
    result = scale(feedback, frame.shape[0], frame.shape[1])

    for keypoint in result['pose'].keys():
        x = result['pose'][keypoint]['x']
        y = result['pose'][keypoint]['y']

        if x >= 0 and y >= 0:
            correct = result['feedback'][keypoint]['correct']
            
            if correct is True:
                cv2.circle(frame, (x, y), 3, user_colors['good'], 2)
            elif correct is None:
                cv2.circle(frame, (x, y), 3, user_colors['undefined'], 2)
            else:
                f_x = result['feedback'][keypoint]['x']
                f_y = result['feedback'][keypoint]['y']
                cv2.circle(frame, (x, y), 3, user_colors['bad'], 2)
                cv2.circle(frame, (f_x, f_y), 3, feedback_color, 2)
    for link in topology:
        keypoint1, keypoint2 = link

        x1 = result['pose'][keypoint1]['x']
        y1 = result['pose'][keypoint1]['y']
        x2 = result['pose'][keypoint2]['x']
        y2 = result['pose'][keypoint2]['y']

        if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
            correct1 = result['feedback'][keypoint1]['correct']
            correct2 = result['feedback'][keypoint2]['correct']

            if correct1 is True and correct2 is True:
                cv2.line(frame, (x1, y1), (x2, y2), user_colors['good'], 2)
            elif correct1 is None and correct2 is None:
                cv2.line(frame, (x1, y1), (x2, y2), user_colors['undefined'], 2)
            else:
                cv2.line(frame, (x1, y1), (x2, y2), user_colors['bad'], 2)
            
                if (correct1 is True or correct1 is None) and (correct2 is False):
                    f_x2 = result['feedback'][keypoint2]['x']
                    f_y2 = result['feedback'][keypoint2]['y']
                    cv2.line(frame, (x1, y1), (f_x2, f_y2), feedback_color, 2)
                elif (correct1 is False) and (correct2 is True or correct2 is None):
                    f_x1 = result['feedback'][keypoint1]['x']
                    f_y1 = result['feedback'][keypoint1]['y']
                    cv2.line(frame, (f_x1, f_y1), (x2, y2), feedback_color, 2)
                else:
                    f_x2 = result['feedback'][keypoint2]['x']
                    f_y2 = result['feedback'][keypoint2]['y']

                    f_x1 = result['feedback'][keypoint1]['x']
                    f_y1 = result['feedback'][keypoint1]['y']

                    cv2.line(frame, (f_x1, f_y1), (f_x2, f_y2), feedback_color, 2)
                    