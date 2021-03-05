from valens import constants
from valens.structures.node import Node
from valens.structures.stream import InputStream, OutputStream, gen_addr_ipc
from valens.structures import sequence

import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
import json
import sys
from valens.dtw import dtw1d
import math
import os
import h5py
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

class DtwKnn:
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        num_features = len(self.X_train)
        num_train = len(self.X_train[0])

        assert len(X_test) == num_features

        f_good = [[] for _ in range(num_features)]
        f_bad = [[] for _ in range(num_features)]

        min_dist = sys.maxsize
        min_dist_path = []
        min_dist_train = -1

        for train in range(num_train):
            for feature in range(num_features):
                _, dist, a1, a2 = dtw1d(X_test[feature], self.X_train[feature][train])
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

def load_features_all(exercise_type, sequences_dir):
    _, _, names = next(os.walk(sequences_dir))
    names = [sequences_dir + '/' + name for name in names if name[0:len(exercise_type)] == exercise_type]
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

def transform(pose, joint_angles, topology, joints):
    '''
    pose: hip, knee, ankle, neck
    joint_angles: ankle, knee, hip, neck [2, 1, 0, 3]
    topology: [knee, ankle], [hip, knee], [neck, hip]
    joints: ankle, knee, hip, neck
    '''
    num_links = len(topology)
    num_joints = num_links + 1

    # calculate link lengths based on pos of joints in pose
    # L = [0, 0.0, 0.7, 0.9]
    L = [0.0]
    for i in range(1, num_joints):
        joint1 = topology[i-1][0]
        joint2 = topology[i-1][1]

        dist = np.linalg.norm(pose[joint1, :] - pose[joint2, :])
        L.append(dist + L[i - 1])

    pose = np.array([
        [pose[2, 0], pose[2, 1] - L[2]],
        [pose[2, 0], pose[2, 1] - L[1]],
        [pose[2, 0], pose[2, 1] - L[0]], # ankle
        [pose[2, 0], pose[2, 1] - L[3]],
    ])

    pose_space = pose.copy()
    pose_space = 1.0 - pose_space
    space_frame = pose_space[2, :].copy()

    pose_space[:, 0] -= space_frame[0]
    pose_space[:, 1] -= space_frame[1]

    # create screw axis for each joint
    # ankle, knee, hip, neck
    Slist = np.empty((6, num_joints))
    for i in range(num_joints):
        Slist[:, i] = [0, 0, 1, L[i], 0, 0]

    corrected_pose = pose.copy()
    for i, joint in enumerate(joints):
        # calculate zero frame of each joint (excluding the ankle) from pose
        M = np.array([
            [1, 0, 0, pose_space[joint, 0]],
            [0, 1, 0, pose_space[joint, 1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # calculate iterative PoE and apply to each joint (excluding the ankle)
        T = mr.FKinSpace(M, Slist[:, 0:i+1], joint_angles[0:i+1])

        # extract pos from each transformation matrix and return the transformed pose
        pos = T[0:2, 3]
        corrected_pose[joint, :] = pos

    # for i in range(1, num_joints):
    #     joint1 = topology[i-1][0]
    #     joint2 = topology[i-1][1]

    #     dist1 = np.linalg.norm(pose[joint1, :] - pose[joint2, :])
    #     dist2 = np.linalg.norm(corrected_pose[joint1, :] - corrected_pose[joint2, :])
    #     print(dist1, dist2)
    # return pose
    corrected_pose[:, 0] += space_frame[0]
    corrected_pose[:, 1] += space_frame[1]
    corrected_pose = 1 - corrected_pose
    return corrected_pose

class Exercise(ABC):
    def __init__(self, exercise_type, sequences_dir=constants.DATA_DIR + '/sequences/post_processed'):
        self.keypoints = [k for k in Keypoints] # change to all keypoints by default
        self.mask = None
        self.topology = None
        self.in_rep = False
        self.initial = None
        self.rep = None
        self.prev_rep = None
        self.x = 0
        self.y = 1
        self.total = 0
        self.t = 0

        self.classifier = DtwKnn()
        X_train, y_train = load_features_all(exercise_type, sequences_dir)

        self.classifier.fit(X_train, y_train)

        self.calc_keypoints()
        self.calc_keypoint_mask()
        self.calc_topology()
        self.calc_angles()
        self.rep = [[] for _ in range(len(self.angles))]

    @abstractmethod
    def project(self):
        pass

    @abstractmethod
    def calc_keypoints(self, pose):
        pass

    def calc_angles(self):
        pass

    def calc_keypoint_mask(self):
        self.mask = np.zeros((len(Keypoints),), dtype=np.bool)
        for keypoint in self.keypoints:
            self.mask[keypoint.value] = 1

    def calc_topology(self, human_pose_path=constants.POSE_JSON):
        with open(human_pose_path, 'r') as f:
            human_pose = json.load(f)

        skeleton = human_pose['skeleton']
        K = len(skeleton)
        self.topology = []
        for k in range(K):
            a = Keypoints(skeleton[k][0] - 1)
            b = Keypoints(skeleton[k][1] - 1)
            if a in self.keypoints and b in self.keypoints:
                self.topology.append([self.keypoints.index(a), self.keypoints.index(b)])

    def predict(self, pose):
        self.total += 1
        if self.initial is None:
            self.initial = pose
            return

        if self.in_rep:
            self.add(pose)
            if self.finished(pose):
                print('finished!', self.total)
                self.eval()
                self.in_rep = False
                self.rep = [[] for _ in range(len(self.angles))]
                self.t = 0
            elif self.prev_rep is not None and self.prev_label == False:
                return np.array([pose[self.mask], self.project(pose[self.mask])])
        elif self.started(pose):
            print('started rep!', self.total)
            if self.mask is None or self.topology is None:
                self.calc_keypoints(pose)
                self.calc_keypoint_mask()
                self.calc_topology()
                self.calc_angles()
                self.rep = [[] for _ in range(len(self.angles))]
            self.add(pose)
            self.in_rep = True

        if self.mask is not None:
            return pose[self.mask]


    def add(self, pose):
        filtered = pose[self.mask]
        body_vecs = calc_body_vecs(filtered, self.topology)
        body_angles = calc_body_angles(body_vecs, self.angles)
        for k in range(len(body_angles)):
            self.rep[k].append(body_angles[k])
        self.t += 1

    def eval(self):
        seq = np.array(self.rep)
        sequence.clean_nans(seq)
        sequence.filter_noise(seq)
        label, alignment, ref_seq = self.classifier.predict(seq)
        print('good' if label else 'bad')
        ref_seq = np.array(ref_seq)
        self.match = align(seq, ref_seq, alignment)
        self.prev_rep = seq
        self.prev_label = label

    @abstractmethod
    def started(self, pose):
        pass

    @abstractmethod
    def finished(self, pose):
        pass

class BodyweightSquat(Exercise):
    def __init__(self, drop_percent=0.90):
        self.hip = 0
        self.knee = 1
        self.ankle = 2
        self.neck = 3
        super().__init__('BS')
        self.drop_percent = drop_percent
        self.prev_hip_y = None

    def calc_keypoints(self, pose=None):
        if pose is None:
            self.keypoints = [Keypoints.RHIP, Keypoints.RKNEE, Keypoints.RANKLE, Keypoints.NECK]
            return

        hip = Keypoints.RHIP.value
        knee = Keypoints.RKNEE.value

        b = (pose[hip, 1] - pose[knee, 1]) / (pose[hip, 0] - pose[knee, 0])
        if b > 0:
            print("detected right side")
            self.keypoints = [Keypoints.RHIP, Keypoints.RKNEE, Keypoints.RANKLE, Keypoints.NECK]
        else:
            print("detected left side")
            self.keypoints = [Keypoints.LHIP, Keypoints.LKNEE, Keypoints.LANKLE, Keypoints.NECK]

    def calc_angles(self):
        self.angles = [[0, 1], [1, 2], [2, 3]]

    def calc_topology(self):
        self.topology = [[self.neck, self.hip], [self.hip, self.knee], [self.knee, self.ankle]]
    
    def started(self, pose):
        return pose[Keypoints.NECK.value, self.y] > (1 - self.drop_percent * (1 - self.initial[Keypoints.NECK.value, self.y]))

    def finished(self, pose):
        f = pose[Keypoints.NECK.value, self.y] <= (1 - self.drop_percent * (1 - self.initial[Keypoints.NECK.value, self.y]))
        if f:
            self.prev_hip_y = None
        return f

    def project(self, pose):
        # shift knee relative to ankle

        current_hip_y = 1 - pose[self.hip, 1]
        # print('hip y:', self.prev_hip_y, current_hip_y)
        if self.prev_hip_y is not None:
            if self.prev_hip_y[0] < current_hip_y and not self.prev_hip_y[1]:
                # print('minimum!')
                self.prev_hip_y[1] = True
            self.prev_hip_y[0] = current_hip_y
        else:
            self.prev_hip_y = [current_hip_y, False]

        lim = 0.1

        neck_to_knee = 0
        hip_to_ankle = 1
        current_hip_to_ankle = self.rep[hip_to_ankle][self.t - 1]
        match_hip_to_ankle = self.match[hip_to_ankle, min(self.match.shape[-1] - 1, self.t - 1)]
        diff_hip_to_ankle = match_hip_to_ankle - current_hip_to_ankle
        if abs(diff_hip_to_ankle / current_hip_to_ankle) < lim:
            diff_hip_to_ankle = 0.0
        current_hip_to_ankle = current_hip_to_ankle + diff_hip_to_ankle * 0.5

        current_neck_to_knee = self.rep[neck_to_knee][self.t - 1]
        match_neck_to_knee = self.match[neck_to_knee, min(self.match.shape[-1] - 1, self.t - 1)]
        diff_neck_to_knee = match_neck_to_knee - current_neck_to_knee
        if abs(diff_neck_to_knee / current_neck_to_knee) < lim:
            diff_neck_to_knee = 0.0
        current_neck_to_knee = current_neck_to_knee + diff_neck_to_knee * 0.5

        current_ankle_to_space = self.rep[2][self.t - 1]
        match_ankle_to_space = self.match[2, min(self.match.shape[-1] - 1, self.t - 1)]
        diff_ankle_to_space = match_ankle_to_space - current_ankle_to_space
        if abs(diff_ankle_to_space / current_ankle_to_space) < lim:
            diff_ankle_to_space = 0.0
        current_ankle_to_space = current_ankle_to_space + diff_ankle_to_space * 0.5

        joint_angles = np.array([current_ankle_to_space , -current_hip_to_ankle, current_neck_to_knee, 0.0])
        corrected_pose = transform(pose, joint_angles, self.topology[::-1], joints=[2, 1, 0, 3])
        corrected_pose[3, :] = pose[3, :]
        return corrected_pose

class FeedbackFilter(Node):
    def __init__(self, pose_address=gen_addr_ipc('pose'), feedback_address=gen_addr_ipc('feedback'), exercise_type=''):
        super().__init__('FeedbackFilter')

        self.input_streams['pose'] = InputStream(pose_address)
        self.output_streams['feedback'] = OutputStream(feedback_address)
        # self.exercise = exercise.load(exercise_type)
        self.exercise = BodyweightSquat()

    def process(self):
        pose = self.input_streams['pose'].recv()
        if pose is None:
            self.stop()
            return

        feedback = self.exercise.predict(pose)
        if feedback is not None:
            self.output_streams['feedback'].send(feedback)
