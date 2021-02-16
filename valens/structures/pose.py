from valens import constants

import  cv2
import json
import numpy as np
import trt_pose.coco

def get_topology(data_path=constants.POSE_JSON):
    with open(data_path, 'r') as f:
        human_pose = json.load(f)
    
    return trt_pose.coco.coco_category_to_topology(human_pose)

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

def scale(p, width, height, round_coords=False):
    s = p.copy()
    s[:, 0] *= width
    s[:, 1] *= height

    if round_coords:
        mask = np.isnan(s)
        s = np.rint(s, out=s).astype(np.int32)
        s[mask] = -1
    return s

def draw_on_image(p, frame, topology, color=(0, 255, 0)):
    s = scale(p, frame.shape[0], frame.shape[1], round_coords=True)
    nan = s == -1
    K = topology.shape[0]
    for k in range(p.shape[0]):
        if not np.any(nan[k, :]):
            cv2.circle(frame, (s[k, 0], s[k, 1]), 3, color, 2)

    for k in range(K):
        c_a = int(topology[k][2])
        c_b = int(topology[k][3])
        if not np.any(nan[c_a, :]) and not np.any(nan[c_b, :]):
            cv2.line(frame, (s[c_a, 0], s[c_a, 1]), (s[c_b, 0], s[c_b, 1]), color, 2)
