from valens import constants
from valens import sequence, exercise
from valens.dtw import dtw1d

import argparse
import h5py
import json
import numpy as np
import os

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import medfilt

def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

def load_features(seq):
    neck_hip_vecs = np.array([(seq[-1, 0, t] - seq[0, 0, t], seq[-1, 1, t] - seq[0, 1, t]) for t in range(seq.shape[-1])])
    hip_knee_vecs = np.array([(seq[0, 0, t] - seq[1, 0, t], seq[0, 1, t] - seq[1, 1, t]) for t in range(seq.shape[-1])])

    neck_hip_vecs = neck_hip_vecs / np.expand_dims(np.linalg.norm(neck_hip_vecs, axis=1), axis=1)
    hip_knee_vecs = hip_knee_vecs / np.expand_dims(np.linalg.norm(hip_knee_vecs, axis=1), axis=1)

    neck_hip_knee_angle = np.degrees(np.arccos(np.clip(np.sum(np.multiply(neck_hip_vecs, hip_knee_vecs), axis=1), -1.0, 1.0)))
    # print(neck_hip_knee_angle)
    return neck_hip_knee_angle
    # return medfilt(medfilt(neck_hip_knee_angle, 5), 5)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a reference sequence from a pose sequence using DTW')
    parser.add_argument('--name', default='', help='Name of .h5 file with user sequences')
    parser.add_argument('--sequences_dir', default=constants.DATA_DIR + '/sequences', help='Directory to store reference sequence')
    parser.add_argument('--outputs_dir', default=constants.DATA_DIR + '/outputs', help='Directory to store output videos')
    parser.add_argument('--fps', default=30, help='Fps of output video')

    args = parser.parse_args()
    
    exercise_type = args.name[0:2]
    e = exercise.load(exercise_type)
    # e.side = exercise.Side.LEFT
    keypoint_mask = e.keypoint_mask()

    ref_filename = args.sequences_dir + '/' + exercise_type + '_ref.h5'
    with h5py.File(ref_filename, 'r') as data:
        ref_seq_i = data["pose"][:]

    user_filename = args.sequences_dir + '/' + args.name + '.h5'
    with h5py.File(user_filename, 'r') as data:
        user_seq = data["pose"][:]
        # reps = data["reps"][:].tolist()

    
    # user_seq = user_seq[keypoint_mask]
    sequence.clean_nans(user_seq)
    sequence.filter_noise(user_seq)

    # rep_idx = sequence.get_rep_idx(reps)
    rep_idx = e.detect_reps(user_seq, drop_percent=0.8)
    reps = sequence.get_reps(user_seq, rep_idx)
    e.detect_side(reps[0])
    exit(0)

    for i in range(len(reps)):
        ref_seq = ref_seq_i.copy()
        user_seq = reps[i]

        # e.normalize(user_seq, ref_seq, axis=1)
        e.center(user_seq)
        e.center(ref_seq)

        user_angle = load_features(user_seq)
        ref_angle = load_features(ref_seq)

        # _, cost, a1, a2 = dtw1d(user_angle, ref_angle)
        cost = DTWDistance(user_angle.tolist(), ref_angle.tolist())
        # cost, path = fastdtw(user_angle, ref_angle, dist=euclidean)
        # a1 = []
        # a2 = []
        # for j in range(len(path)):
        #     a1.append(path[j][0])
        #     a2.append(path[j][1])
        # alignment = [a1, a2]
        # alignments = [alignment]*4
        print(cost)

        # # alignments, cost = sequence.warp(user_angle, ref_angle)
        # aligned_user_seq = sequence.align(user_seq, ref_seq, alignments)
        
        # output_filename = args.outputs_dir + '/' + args.name + '_r' + str(i) + '_aligned.mp4'
        # print(output_filename, cost)
        # sequence.visualize([user_seq, aligned_user_seq], [(255, 255, 255), (0, 255, 0)], output_filename, e.topology(), fps=int(args.fps))
