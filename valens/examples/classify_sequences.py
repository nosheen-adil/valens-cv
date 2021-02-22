from valens import constants
from valens.structures import sequence, exercise

import argparse
import h5py
import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import math
import time
import sys

from valens.dtw import dtw1d

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
    hip = 0
    knee = 1
    ankle = 2
    neck = 3

    x = 0
    y = 1

    neck_hip_vecs = np.array([(seq[neck, x, t] - seq[hip, x, t], seq[neck, y, t] - seq[hip, y, t]) for t in range(seq.shape[-1])])
    hip_knee_vecs = np.array([(seq[hip, x, t] - seq[knee, x, t], seq[hip, y, t] - seq[knee, y, t]) for t in range(seq.shape[-1])])
    knee_ankle_vecs = np.array([(seq[knee, x, t] - seq[ankle, x, t], seq[knee, y, t] - seq[ankle, y, t]) for t in range(seq.shape[-1])])

    neck_hip_vecs = neck_hip_vecs / np.expand_dims(np.linalg.norm(neck_hip_vecs, axis=1), axis=1)
    hip_knee_vecs = hip_knee_vecs / np.expand_dims(np.linalg.norm(hip_knee_vecs, axis=1), axis=1)
    knee_ankle_vecs = knee_ankle_vecs / np.expand_dims(np.linalg.norm(knee_ankle_vecs, axis=1), axis=1)

    neck_hip_knee_angle = np.degrees(np.arccos(np.clip(np.sum(np.multiply(neck_hip_vecs, hip_knee_vecs), axis=1), -1.0, 1.0)))
    hip_knee_ankle_angle = np.degrees(np.arccos(np.clip(np.sum(np.multiply(hip_knee_vecs, knee_ankle_vecs), axis=1), -1.0, 1.0)))
    return neck_hip_knee_angle, hip_knee_ankle_angle

def load_features_all(names):
    features1 = []
    features2 = []
    labels = []

    for name in names:
        with h5py.File(name, 'r') as data:
            seq = data['pose'][:]
            label = data['label'][()]

        f1, f2 = load_features(seq)
        features1.append(f1)
        features2.append(f2)
        labels.append(label)

    return features1, features2, labels

def save_alignment(ref_name, user_name, alignment, filename, e):
    with h5py.File(ref_name, 'r') as data:
        ref_seq = data['pose'][:]
    with h5py.File(user_name, 'r') as data:
        user_seq = data['pose'][:]
    aligned_seq = sequence.align(user_seq, ref_seq, [alignment]*ref_seq.shape[0])
    sequence.visualize([user_seq, aligned_seq], [(0, 255, 0), (255, 255, 255)], filename, e.topology())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a reference sequence from a pose sequence using DTW')
    parser.add_argument('--exercise', default='', help='Short-form of exercise')
    parser.add_argument('--sequences_dir', default=constants.DATA_DIR + '/sequences/post_processed', help='Directory to store reference sequence')
    parser.add_argument('--outputs_dir', default=constants.DATA_DIR + '/outputs/aligned', help='Directory to store output videos')
    parser.add_argument('--fps', default=30, help='Fps of output video')

    args = parser.parse_args()
    
    exercise_type = args.exercise
    e = exercise.load(exercise_type)

    _, _, input_filenames = next(os.walk(args.sequences_dir))
    input_filenames = [args.sequences_dir + '/' + input_filename for input_filename in input_filenames if input_filename[0:len(exercise_type)] == exercise_type]
    X_train_names, X_test_names = train_test_split(input_filenames, test_size=0.4)
    # print(X_train_names)
    # print(X_test_names)
    X_train1, X_train2, y_train = load_features_all(X_train_names)
    X_test1, X_test2, y_test = load_features_all(X_test_names)
    
    # print(X_train, y_train)

    predictions = []
    for test in range(len(X_test_names)):
        start = time.time()
        f1_good, f2_good, f1_bad, f2_bad = [[] for _ in range(4)]
        min_dist = sys.maxsize
        min_dist_path = []
        min_dist_train = -1
        for train in range(len(X_train_names)):
            # dist1 = DTWDistance(X_train1[train], X_test1[test])
            # dist2 = DTWDistance(X_train2[train], X_test2[test])
            # start = time.time()
            _, dist1, a11, a12 = dtw1d(X_test1[test], X_train1[train])
            _, dist2, a21, a22 = dtw1d(X_test2[test], X_train2[train])
            dist1 = math.sqrt(dist1)
            dist2 = math.sqrt(dist2)

            if y_train[train]:
                print("good", X_train_names[train])
                if dist1 < min_dist:
                    min_dist = dist1
                    min_dist_path = [a11, a12]
                    min_dist_train = train
                if dist2 < min_dist:
                    min_dist = dist2
                    min_dist_path = [a21, a22]
                    min_dist_train = train
            # end = time.time()
            # print("FPS", 1/(end-start))
            if y_train[train]:
                f1_good.append(dist1)
                f2_good.append(dist2)
            else:
                f1_bad.append(dist1)
                f2_bad.append(dist2)

        print(min_dist, min_dist_path)
        good_score = np.mean(f1_good) + np.mean(f2_good)
        bad_score = np.mean(f1_bad) + np.mean(f2_bad)

        if good_score < bad_score:
            predictions.append(1)
        else:
            predictions.append(0)

        save_alignment(X_train_names[min_dist_train], X_test_names[test], min_dist_path, args.outputs_dir + '/' + os.path.splitext(os.path.basename(X_test_names[test]))[0] + '.mp4', e)
        end = time.time()
        print("Fps:", 1/(end-start))

    # print(predictions, y_test)
    print(classification_report(y_test, predictions, target_names=['correct', 'bad']))
    
