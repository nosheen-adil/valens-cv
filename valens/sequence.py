from valens import constants
import valens as va
from valens.dtw import dtw1d

import os
import numpy as np
import h5py
import sys
import math
from scipy import ndimage

def filter_noise(seq, size=5):
    for k in range(seq.shape[0]):
        if len(seq.shape) == 3:
            for i in range(2):
                seq[k, i, :] = ndimage.median_filter(seq[k, i, :], size)
        else:
            seq[k, :] = ndimage.median_filter(seq[k, :], size)

def load_features(filenames):
    features = []
    labels = []

    for filename in filenames:
        with h5py.File(filename, 'r') as data:
            seq = data['seq'][:]
            label = data['label'][()]

        features.append(seq)
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

class DtwKnn:
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        num_train = len(self.X_train)
        num_features = self.X_train[0].shape[0]

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

        return y_test, align(X_test, self.X_train[min_dist_train], min_dist_path)
        
def match_frame(x, j, val):
    n = x.shape[0]
    curr_dist = abs(x[j] - val)
    while j < (n - 1):
        dist = abs(x[j+1] - val)
        if dist <= curr_dist:
            j += 1
            curr_dist = dist
        else:
            break
    return j