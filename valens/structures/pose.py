from valens import constants

import  cv2
import json
import numpy as np
import trt_pose.coco
from enum import Enum
import modern_robotics as mr
import sys

class Keypoints(Enum):
    NOSE = 0
    LEYE = 1
    REYE = 2
    LEAR = 3
    REAR = 4
    LSHOULDER = 5
    RSHOULDER = 6
    LELBOW = 7
    RELBOW = 8
    LWRIST = 9
    RWRIST = 10
    LHIP = 11
    RHIP = 12
    LKNEE = 13
    RKNEE = 14
    LANKLE = 15
    RANKLE = 16
    NECK = 17

    def __str__(self):
        names = [
            'neck', 
            'left_eye',
            'right_eye',
            'left_ear',
            'right_ear',
            'left_shoulder',
            'right_shoulder',
            'left_elbow',
            'right_elbow',
            'left_wrist',
            'right_wrist',
            'left_hip',
            'right_hip',
            'left_knee',
            'right_knee',
            'left_ankle',
            'right_ankle',
            'neck'
        ]
        return names[self.value]

def topology(keypoints=[], human_pose_path=constants.POSE_JSON):
    if not keypoints:
        keypoints = [k for k in Keypoints]

    with open(human_pose_path, 'r') as f:
        human_pose = json.load(f)

    skeleton = human_pose['skeleton']
    K = len(skeleton)
    topology = []
    for k in range(K):
        a = Keypoints(skeleton[k][0] - 1)
        b = Keypoints(skeleton[k][1] - 1)

        if a in keypoints and b in keypoints:
            topology.append([a.value, b.value])

    return topology

def filter_keypoints(pose, keypoints):
    num_joints = len(keypoints)
    filtered = np.empty((num_joints, 2))
    for i, keypoint in enumerate(keypoints):
        filtered[i, :] = pose[keypoint.value, :]
    return filtered 

def _nn_find_person_idx(object_counts, objects, min_keypoints, prev):
    count = int(object_counts[0])
    results = {}
    idx = []
    # print("count: ", count)
    for i in range(count):
        obj = objects[0][i]
        C = obj.shape[0]
        non_neg = 0
        for j in range(C):
            k = int(obj[j])
            if k >= 0:
                non_neg += 1
        if non_neg >= min_keypoints:
            # print(non_neg)
            idx.append(i)

    return idx

def _nn_to_numpy(data, person_idx, objects, normalized_peaks):
    C = 18
    obj = objects[0][person_idx]
    for j in range(C):
        k = int(obj[j])
        if k >= 0:
            peak = normalized_peaks[0][j][k]
            x = float(peak[1])
            y = float(peak[0])
            data[j, 0] = x
            data[j, 1] = y

def nn_to_numpy(object_counts, objects, normalized_peaks, min_keypoints=7, prev=None, center=[0.5, 0.5]):
    C = 18
    data = np.empty((18, 2))
    data[:] = np.nan

    idx = _nn_find_person_idx(object_counts, objects, min_keypoints, prev)
    num_persons = len(idx)
    if num_persons == 0:
        return data
    elif num_persons == 1:
        person_idx = idx[0]
    else:
        print("Warning:", len(idx), "objects detected")

        min_dist = sys.maxsize
        min_dist_idx = -1
        data = np.tile(data, (num_persons, 1, 1))
        if prev is None:
            print("No previous pose to match")
            hip = Keypoints.RHIP.value
            for i, person_idx in enumerate(idx):
                _nn_to_numpy(data[i, :, :], person_idx, objects, normalized_peaks)
                dist = np.linalg.norm(center - data[i, hip, :])
                if dist < min_dist:
                    min_dist = dist
                    min_dist_idx = i
        else:
            prev_mask = np.logical_not(np.isnan(prev))
            for i, person_idx in enumerate(idx):
                _nn_to_numpy(data[i, :, :], person_idx, objects, normalized_peaks)
                data_mask = np.logical_and(np.logical_not(np.isnan(data[i, :, :])), prev_mask)
                dist = np.sum(np.linalg.norm(prev[data_mask] - data[i, :, :][data_mask]))
                dist += abs(np.count_nonzero(data_mask) - np.count_nonzero(prev_mask)) * 1000
                if dist < min_dist:
                    min_dist = dist
                    min_dist_idx = i
        print("Found match to prev pose with person:", person_idx)
        return data[min_dist_idx, :, :]

    _nn_to_numpy(data, person_idx, objects, normalized_peaks)
    return data

def scale(pose, width=constants.POSE_MODEL_WIDTH, height=constants.POSE_MODEL_HEIGHT, round_coords=False):
    s = pose.copy()
    s[:, 0] *= width
    s[:, 1] *= height

    if round_coords:
        mask = np.isnan(s)
        s = np.rint(s, out=s).astype(np.int32)
        s[mask] = -1
    return s

def draw_on_image(pose, frame, topology, color=(0, 255, 0)):
    s = scale(pose, frame.shape[0], frame.shape[1], round_coords=True)
    nan = s == -1
    K = len(topology)

    for k in range(pose.shape[0]):
        if not np.any(nan[k, :]):
            cv2.circle(frame, (s[k, 0], s[k, 1]), 3, color, 2)

    for k in range(K):
        c_a = topology[k][0]
        c_b = topology[k][1]

        if not np.any(nan[c_a, :]) and not np.any(nan[c_b, :]):
            cv2.line(frame, (s[c_a, 0], s[c_a, 1]), (s[c_b, 0], s[c_b, 1]), color, 2)

def vectors(pose):
    num_vecs = pose.shape[0] - 1
    vectors = np.empty((num_vecs, 2))

    for v in range(num_vecs):
        for i in range(2):
            joint1 = v
            joint2 = v + 1
            vectors[v, i] =  np.array([pose[joint2, i] - pose[joint1, i]])
        vectors[v] /= np.linalg.norm(vectors[v])
    return vectors

def angles(pose, space_frame=[0, 1]):
    vecs = vectors(pose)
    if space_frame is not None:
        vecs = np.vstack((space_frame, vecs))
    num_vecs = vecs.shape[0]
    angles = np.empty((vecs.shape[0] - 1,))
    for c in range(angles.shape[0]):
        vec1 = c
        vec2 = c + 1
        angles[c] = np.arccos(vecs[vec2, 0] * vecs[vec1, 0] + vecs[vec2, 1] * vecs[vec1, 1])
    return angles

def transform(zero_pose, Slist, joint_angles):
    num_joints = zero_pose.shape[0]
    num_links = num_joints - 1
    assert len(joint_angles) == num_joints

    # calculate link lengths based on pos of joints in pose
    L = [0.0]
    for i in range(1, num_joints):
        joint1 = i - 1
        joint2 = i

        dist = np.linalg.norm(zero_pose[joint2, :] - zero_pose[joint1, :])
        L.append(dist + L[i - 1])

    _zero_pose = zero_pose.copy()
    space_frame = zero_pose[0, :].copy()
    _zero_pose -= space_frame

    # create screw axis for each joint
    # ankle, knee, hip, neck
    # Slist = np.empty((6, num_joints))
    # for i in range(num_joints):
    #     Slist[:, i] = [0, 0, 1, L[i], 0, 0]

    pose = np.empty((num_joints, 2))
    for i in range(num_joints):
        # calculate zero frame of each joint (excluding the ankle) from pose
        M = np.array([
            [1, 0, 0, _zero_pose[i, 0]],
            [0, 1, 0, _zero_pose[i, 1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # calculate iterative PoE and apply to each joint (excluding the ankle)
        T = mr.FKinSpace(M, Slist[:, 0:i+1], joint_angles[0:i+1])

        # extract pos from each transformation matrix and return the transformed pose
        pos = T[0:2, 3]
        pose[i, :] = pos

    pose += space_frame
    return pose

def reflect(pose):
    for i in range(pose.shape[0]):
        pose[i, 0] = 1 - pose[i, 0]
