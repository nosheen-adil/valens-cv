from valens import constants
from valens.structures import pose
from valens.dtw import dtw2d

import cv2
#from pydtw import dtw2d, dtw1d
import json
import numpy as np
import os
import time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy import ndimage

all_keypoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"]

def get_rep_idx(markers):
    rep_idx = np.empty((len(markers) + 1,2), dtype=np.uint32)
    markers.insert(0, 0.0)
    markers.append(-1)
    for i in range(len(markers) - 1):
        rep_idx[i, :] = [markers[i], markers[i+1]]
    return rep_idx

def get_reps(p, rep_idx):
    refs = []
    for i in range(rep_idx.shape[0]):
        refs.append(p[:, :, rep_idx[i, 0]:rep_idx[i, 1]])
    return refs

def gen_keypoints_mask(keypoints=[]):
    with open(constants.POSE_JSON, 'r') as f:
        human_pose = json.load(f)

    mask = np.zeros((len(all_keypoints,)), dtype=np.bool)
    if not keypoints:
        mask[:] = 1
        return mask

    for keypoint in keypoints:
        mask[keypoint.value] = 1
    return mask

def calc_center(seq):
    pose = seq[:, :, 0]
    # right = all_keypoints.index("right_hip")
    # left = all_keypoints.index("left_hip")
    # center = (pose[right, :] + pose[left, :]) / 2
    hip = all_keypoints.index("right_hip")
    center = pose[hip, :]
    return center

def center(seq, width=1, height=1):
    global_center = np.array([width / 2, height / 2])
    global_center[1] -= 0.1
    center = calc_center(seq)
    diff = global_center - center
    for i in range(2):
        seq[:, i, :] += diff[i]

def visualize(seqs, colors, filename, topology, capture=None, fps=10, width=constants.POSE_MODEL_WIDTH, height=constants.POSE_MODEL_HEIGHT):
    if os.path.exists(filename):
        os.remove(filename)
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    total_frames = seqs[0].shape[-1]
    for i in range(total_frames):
        if capture:
            ret, image = capture.read()
            image = cv2.resize(image, (width, height))
            assert(ret)
        else:
            image = np.zeros((width, height, 3), dtype=np.uint8)
        for j in range(len(seqs)):
            pose.draw_on_image(seqs[j][:, :, i], image, topology, color=colors[j])
        writer.write(image)

def clean_nans(seq):
    x = lambda z: z.nonzero()[0]

    for k in range(seq.shape[0]):
        for i in range(2):
            nans = np.isnan(seq[k, i, :])
            if nans.all():
                continue
            seq[k, i, nans] = np.interp(x(nans), x(~nans), seq[k, i, ~nans])

def filter_noise(seq, size=5):
    for k in range(seq.shape[0]):
        for i in range(2):
            seq[k, i, :] = ndimage.median_filter(seq[k, i, :], size)

def filter_keypoints(seq, keypoints):
    mask = gen_keypoints_mask(keypoints)
    return seq[mask, :, :]

def warp(seq1, seq2):
    assert seq1.shape[0] == seq2.shape[0]
    start = time.time()
    total_keypoints = seq1.shape[0]
    alignments = []
    cost = np.empty((total_keypoints,), dtype=np.float32)
    for k in range(total_keypoints):
        x1 = np.transpose(seq1[k, :, :])
        x2 = np.transpose(seq2[k, :, :])
        path = []
        if not np.isnan(x1).any() and not np.isnan(x2).any():
            distance, path = fastdtw(x1, x2, dist=euclidean)
            cost[k] = distance

        # x1 = np.transpose(seq1[k, :, :]).copy(order='c')
        # x2 = np.transpose(seq2[k, :, :]).copy(order='c')
        # _, c, a1, a2 = dtw2d(x1, x2)
        # cost[k] = c
        a1 = []
        a2 = []
        for i in range(len(path)):
            a1.append(path[i][0])
            a2.append(path[i][1])
        
        # print(c)
        # alignment = np.transpose(np.array([a1, a2]))
        alignment = [a1, a2]
        alignments.append(alignment)
    end = time.time()
    print("seq_dtw fps:", 1/(end-start))
    return alignments, cost

def align(seq1, seq2, alignments):
    assert seq1.shape[0] == seq2.shape[0]
    assert seq1.shape[0] == len(alignments)

    total_keypoints = len(alignments)
    align_seq2 = np.empty_like(seq1) 
    align_seq2[:] = np.nan
    for k in range(total_keypoints):
        alignment = alignments[k]
        total_warped_frames = len(alignment[0])
        
        w = 0 # counter in alignment[0] (matched frames in seq1)
        frame = 0 # frame number in seq1
        total_vals = 0
        val = np.zeros((2, ))
        while w < total_warped_frames:
            if alignment[0][w] == frame:
                # cumulate results
                val += seq2[k, :, alignment[1][w]]
                total_vals += 1
                w += 1
            else:
                # write to output
                if total_vals > 0:
                    align_seq2[k, :, frame] = (val / total_vals) # avg seq1 matches in seq2
                    val[:] = 0.0
                    total_vals = 0
                # else:
                    # print("skipping!!!", k)
                frame += 1
        if total_vals > 0: # last frame
            align_seq2[k, :, frame] = (val / total_vals) # avg seq1 matches in seq2

    return align_seq2

def merge(seq1, seq2):
    assert seq1.shape == seq2.shape
    merged_seq = np.stack((seq1, seq2), axis=3)
    return np.mean(merged_seq, axis=3)
