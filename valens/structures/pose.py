from valens import constants

import  cv2
import json
import numpy as np
import trt_pose.coco
from enum import Enum
import modern_robotics as mr

class Keypoints(Enum):
    NOSE = 0
    REYE = 1
    LEYE = 2
    REAR = 3
    LEAR = 4
    RSHOULDER = 5
    LSHOULDER = 6
    RELBOW = 7
    LELBOW = 8
    RWRIST = 9
    LWRIST = 10
    RHIP = 11
    LHIP = 12
    RKNEE = 13
    LKNEE = 14
    RANKLE = 15
    LANKLE = 16
    NECK = 17

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
            topology.append([keypoints.index(a), keypoints.index(b)])

    return topology

def keypoint_mask(keypoints):
    mask = np.zeros((len(Keypoints),), dtype=np.bool)
    for keypoint in keypoints:
        mask[keypoint.value] = 1
    return mask

def _nn_find_person_idx(object_counts, objects, min_keypoints):
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
    if len(idx) > 1:
        print("Warning:", len(idx), "objects detected")
    if len(idx) == 0:
        return -1
    return idx[0]

def nn_to_numpy(object_counts, objects, normalized_peaks, min_keypoints=10):
    idx = _nn_find_person_idx(object_counts, objects, min_keypoints)
    obj = objects[0][idx]
    C = obj.shape[0]
    data = np.empty((C, 2))
    data[:] = np.nan

    if idx < 0:
        return data

    for j in range(C):
        k = int(obj[j])
        if k >= 0:
            peak = normalized_peaks[0][j][k]
            x = float(peak[1])
            y = float(peak[0])
            data[j, 0] = x
            data[j, 1] = y

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

def vectors(pose, topology):
    num_vecs = len(topology)
    vectors = np.empty((num_vecs, 2))

    for v in range(num_vecs):
        for i in range(2):
            vectors[v, i] =  np.array([pose[topology[v][1], i] - pose[topology[v][0], i]])
        vectors[v] /= np.linalg.norm(vectors[v])
    return vectors

def angles(pose, topology, space_frame=[0, 1]):
    vecs = vectors(pose, topology)
    vecs = np.vstack((space_frame, vecs))

    num_vecs = vecs.shape[0]
    angles = np.empty((len(topology),))
    for c in range(len(topology)):
        joint1 = topology[c][0]
        joint2 = topology[c][1]
        angles[c] = np.arccos(vecs[joint2, 0] * vecs[joint1, 0] + vecs[joint2, 1] * vecs[joint1, 1])
    return angles


def transform(ref_pose, joint_angles, topology):
    num_links = len(topology)
    num_joints = num_links + 1
    assert len(joint_angles) == num_joints
    assert ref_pose.shape[0]

    # calculate link lengths based on pos of joints in pose
    L = [0.0]
    for i in range(1, num_joints):
        joint1 = topology[i-1][0]
        joint2 = topology[i-1][1]

        dist = np.linalg.norm(ref_pose[joint2, :] - ref_pose[joint1, :])
        L.append(dist + L[i - 1])

    zero_pose = np.empty((num_joints, 2))
    for i in range(num_joints):
        zero_pose[i, :] = [ref_pose[0, 0], ref_pose[0, 1] - L[i]]

    space_frame = zero_pose[0, :].copy()
    zero_pose -= space_frame

    # create screw axis for each joint
    # ankle, knee, hip, neck
    Slist = np.empty((6, num_joints))
    for i in range(num_joints):
        Slist[:, i] = [0, 0, 1, -L[i], 0, 0]

    pose = np.empty((num_joints, 2))
    for i in range(num_joints):
        # calculate zero frame of each joint (excluding the ankle) from pose
        M = np.array([
            [1, 0, 0, zero_pose[i, 0]],
            [0, 1, 0, zero_pose[i, 1]],
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
