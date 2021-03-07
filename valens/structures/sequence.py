from valens import constants
from valens import structures as core
from valens.dtw import dtw1d

import os
import numpy as np
import h5py
import sys
import math

class DtwKnn:
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        num_train = len(self.X_train)
        num_features = self.X_train[0].shape[0]
        print(num_train, num_features)

        assert X_test.shape[0] == num_features

        f_good = [[] for _ in range(num_features)]
        f_bad = [[] for _ in range(num_features)]

        min_dist = sys.maxsize
        min_dist_path = []
        min_dist_train = -1

        for train in range(num_train):
            for feature in range(num_features):
                _, dist, a1, a2 = dtw1d(X_test[feature, :], self.X_train[train][feature, :])
                dist = math.sqrt(dist)

                if self.y_train[train]:
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_path = [a1, a2]
                        min_dist_train = train

                if self.y_train[train]:
                    f_good[feature].append(dist)
                else:
                    f_bad[feature].append(dist)
        print('min dist', min_dist)

        good_score = np.sum(np.mean(f_good, axis=0))
        bad_score = np.sum(np.mean(f_bad, axis=0))

        if good_score < bad_score:
            y_test = 1
        else:
            y_test = 0

        return y_test

def angles(seq, topology, space_frame=[0, 1]):
    T = seq.shape[-1]
    num_vecs = len(topology) + 1 # for space frame
    angles = np.empty((num_vecs - 1, T))
    for t in range(T):
        angles[:, t] = core.pose.angles(seq[:, :, t], topology)
    return angles

def load_features(filenames, topology):
    features = []
    labels = []

    for filename in filenames:
        with h5py.File(filename, 'r') as data:
            seq = data['pose'][:]
            label = data['label'][()]

        features.append(angles(seq, topology))
        labels.append(label)

    return features, np.array(labels, dtype=np.bool)

def post_processed_filenames(exercise_type, sequences_dir=constants.DATA_DIR + '/sequences/post_processed'):
    _, _, names = next(os.walk(sequences_dir))
    names = [sequences_dir + '/' + name for name in names if name[0:len(exercise_type)] == exercise_type]
    return names

def align(seq1, seq2, alignment):
    assert seq1.shape[0] == seq2.shape[0]

    total_keypoints = seq1.shape[0]
    align_seq2 = np.empty_like(seq1) 
    align_seq2[:] = np.nan
    for k in range(total_keypoints):
        total_warped_frames = len(alignment[0])
        
        w = 0 # counter in alignment[0] (matched frames in seq1)
        frame = 0 # frame number in seq1
        total_vals = 0
        val = 0.0
        while w < total_warped_frames:
            if alignment[0][w] == frame:
                # cumulate results
                val += seq2[k, alignment[1][w]]
                total_vals += 1
                w += 1
            else:
                # write to output
                if total_vals > 0:
                    align_seq2[k, frame] = (val / total_vals) # avg seq1 matches in seq2
                    val = 0.0
                    total_vals = 0
                # else:
                    # print("skipping!!!", k)
                frame += 1
        if total_vals > 0: # last frame
            align_seq2[k, frame] = (val / total_vals) # avg seq1 matches in seq2

    return align_seq2
    