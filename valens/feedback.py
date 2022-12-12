import valens as va
from valens.pose import Keypoints

import cv2
from enum import Enum
import math
import numpy as np

class KeypointResult(Enum):
    GOOD = 'good'
    BAD = 'bad'
    UNDEFINED = 'undefined'

def to_json(user_pose, labels, corrected_pose=None, keypoints=[]):
    if corrected_pose is not None:
        assert user_pose.shape == corrected_pose.shape

    if not keypoints:
        keypoints = [k for k in Keypoints]
    else:
        assert user_pose.shape[0] == len(keypoints)
    keypoint_names = [str(k) for k in keypoints]

    if corrected_pose is None:
        result = {
            'pose' : {}
        }
    else:
        result = {
            'pose' : {},
            'feedback' : {}
        }

    for i, keypoint_name in enumerate(keypoint_names):
        label = labels[i]
        result['pose'][keypoint_name] = {
            'x' : user_pose[i, 0],
            'y' : user_pose[i, 1]
        }

        if label is KeypointResult.GOOD:
            result['feedback'][keypoint_name] = {
                'correct' : True
            }
        elif label is KeypointResult.BAD:
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
    

def scale(feedback, width=va.constants.POSE_MODEL_WIDTH, height=va.constants.POSE_MODEL_HEIGHT):
    result = feedback.copy()
    for keypoint in result['pose'].keys():
        if math.isnan(result['pose'][keypoint]['x']):
            result['pose'][keypoint]['x'] = -1
        else:
            result['pose'][keypoint]['x'] = round(result['pose'][keypoint]['x'] * width)

        if math.isnan(result['pose'][keypoint]['y']):
            result['pose'][keypoint]['y'] = -1
        else:
            result['pose'][keypoint]['y'] = round(result['pose'][keypoint]['y'] * height)

    if 'feedback' in result:
        for keypoint in result['feedback'].keys():
            label = result['feedback'][keypoint]['correct']
            if label is False:
                if math.isnan(result['feedback'][keypoint]['x']):
                    result['feedback'][keypoint]['x'] = -1
                else:
                    result['feedback'][keypoint]['x'] = round(result['feedback'][keypoint]['x'] * width)
                if math.isnan(result['feedback'][keypoint]['y']):
                    result['feedback'][keypoint]['y'] = -1
                else:
                    result['feedback'][keypoint]['y'] = round(result['feedback'][keypoint]['y'] * height)

    return result

def draw_on_image(feedback, frame, topology, user_colors={True : (0, 153, 0), False : (0, 0, 204), None : (255, 255, 255)}, feedback_color=(0, 153, 0), fps=0):
    result = scale(feedback, frame.shape[1], frame.shape[0])
    thickness = 2
    for keypoint in result['pose'].keys():
        x = result['pose'][keypoint]['x']
        y = result['pose'][keypoint]['y']

        if x >= 0 and y >= 0:
            if 'feedback' in feedback:
                label = result['feedback'][keypoint]['correct']
                cv2.circle(frame, (x, y), 3, user_colors[label], thickness)
                if label is False:
                    f_x = result['feedback'][keypoint]['x']
                    f_y = result['feedback'][keypoint]['y']
                    cv2.circle(frame, (f_x, f_y), 3, feedback_color, thickness)
            else:
                cv2.circle(frame, (x, y), 3, user_colors[None], thickness)

    for link in topology:
        keypoint1, keypoint2 = link

        x1 = result['pose'][keypoint1]['x']
        y1 = result['pose'][keypoint1]['y']
        x2 = result['pose'][keypoint2]['x']
        y2 = result['pose'][keypoint2]['y']

        if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
            if 'feedback' in feedback:
                label1 = result['feedback'][keypoint1]['correct']
                label2 = result['feedback'][keypoint2]['correct']

                if label1 is True and label2 is True:
                    cv2.line(frame, (x1, y1), (x2, y2), user_colors[True], thickness)
                else:
                    cv2.line(frame, (x1, y1), (x2, y2), user_colors[False], thickness)
                
                    if label1 is True and label2 is False:
                        f_x2 = result['feedback'][keypoint2]['x']
                        f_y2 = result['feedback'][keypoint2]['y']
                        cv2.line(frame, (x1, y1), (f_x2, f_y2), feedback_color, thickness)
                    elif label1 is False and label2 is True:
                        f_x1 = result['feedback'][keypoint1]['x']
                        f_y1 = result['feedback'][keypoint1]['y']
                        cv2.line(frame, (f_x1, f_y1), (x2, y2), feedback_color, thickness)
                    else:
                        f_x2 = result['feedback'][keypoint2]['x']
                        f_y2 = result['feedback'][keypoint2]['y']

                        f_x1 = result['feedback'][keypoint1]['x']
                        f_y1 = result['feedback'][keypoint1]['y']

                        cv2.line(frame, (f_x1, f_y1), (f_x2, f_y2), feedback_color, thickness)
            else:
                cv2.line(frame, (x1, y1), (x2, y2), user_colors[None], thickness)       
    if fps > 0:
        cv2.putText(
            frame,
            'Pipeline Performance: {0} FPS'.format(fps), 
            (30, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5,
            (255, 255, 255),
            2
        )

