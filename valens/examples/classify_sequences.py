from valens import constants
from valens import sequence, exercise

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

class DtwKnn:
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        num_features = len(self.X_train)
        num_train = len(self.X_train[0])
        # print(num_train, num_features, len(X_test))
        assert len(X_test) == num_features

        f_good = [[] for _ in range(num_features)]
        f_bad = [[] for _ in range(num_features)]

        min_dist = sys.maxsize
        min_dist_path = []
        min_dist_train = -1

        for train in range(num_train):
            for feature in range(num_features):
                _, dist, a1, a2 = dtw1d(X_test[feature].copy('c'), self.X_train[feature][train].copy('c'))
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

        return y_test, min_dist_path, [self.X_train[feature][min_dist_train].copy() for feature in range(num_features)]

def load_features(seq):
    hip = 0
    knee = 1
    ankle = 2
    neck = 3

    x = 0
    y = 1

    T = seq.shape[-1]
    topology = [[neck, hip], [hip, knee], [knee, ankle]]
    body_angles = np.empty((T,3))
    for t in range(T):
        body_vecs = calc_body_vecs(seq[:, :, t], topology)
        body_angles[t, :] = calc_body_angles(body_vecs, connections=[[0, 1], [1, 2], [2, 3]])
    
    return body_angles[:, 0], body_angles[:, 1], body_angles[:, 2]

def load_features_all(names):
    features = [[] for _ in range(3)]
    labels = []

    for name in names:
        with h5py.File(name, 'r') as data:
            seq = data['pose'][:]
            label = data['label'][()]

        f1, f2, f3 = load_features(seq)
        features[0].append(f1)
        features[1].append(f2)
        features[2].append(f3)
        labels.append(label)

    return features, np.array(labels, dtype=np.bool)

def calc_body_vecs(pose, topology):
    p = 1 - pose
    p[:, 0] -= p[2, 0]
    p[:, 1] -= p[2, 1]

    num_vecs = len(topology)
    body_vecs = np.empty((num_vecs + 1, 2))

    for v in range(num_vecs):
        for i in range(2):
            body_vecs[v, i] =  np.array([p[topology[v][0], i] - p[topology[v][1], i]])
        body_vecs[v] /= np.linalg.norm(body_vecs[v])
    body_vecs[-1, :] = [0, 1]
    return body_vecs

def calc_body_angles(body_vecs, connections=[]):
    num_vecs = body_vecs.shape[0]
    if not connections:
        connections = [[i, i+1] for i in range(num_vecs - 1)]
    body_angles = np.empty((len(connections),))
    for c in range(len(connections)):
        joint1 = connections[c][0]
        joint2 = connections[c][1]
        body_angles[c] = np.arccos(body_vecs[joint1, 0] * body_vecs[joint2, 0] + body_vecs[joint1, 1] * body_vecs[joint2, 1])
    return body_angles

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
    X_train, y_train = load_features_all(X_train_names)
    X_test, y_test = load_features_all(X_test_names)
    # print(X_test[0].shape)
    # print(X_train, y_train)

    classifier = DtwKnn()
    classifier.fit(X_train, y_train)

    predictions = []
    for test in range(len(X_test_names)):
        start = time.time()
        x_test = [X_test[feature][test] for feature in range(len(X_test))]
        label, alignment, ref_seq = classifier.predict(x_test)
        # save_alignment(X_train_names[min_dist_train], X_test_names[test], min_dist_path, args.outputs_dir + '/' + os.path.splitext(os.path.basename(X_test_names[test]))[0] + '.mp4', e)
        predictions.append(label)
        end = time.time()
        print("Fps:", 1/(end-start))

    # print(predictions, y_test)
    print(classification_report(y_test, predictions, target_names=['correct', 'bad']))
    
