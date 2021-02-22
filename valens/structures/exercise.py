from valens import constants

from abc import ABC, abstractmethod
from enum import Enum
import json
import numpy as np

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

class ExerciseType(Enum):
    BS = "BS"

class Side(Enum):
    RIGHT = 0
    LEFT = 1

class Exercise(ABC):
    @abstractmethod
    def keypoints(self):
        pass

    def keypoint_mask(self):
        keypoints = self.keypoints()
        mask = np.zeros((len(Keypoints),), dtype=np.bool)
        for keypoint in keypoints:
            mask[keypoint.value] = 1
        return mask

    def topology(self, human_pose_path=constants.POSE_JSON):
        keypoints = self.keypoints()
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

    @abstractmethod
    def compute_center(self, pose):
        pass
    
    @abstractmethod
    def center(self, seq, width=1, height=1):
        pass

    @abstractmethod
    def normalize(self, seq1, seq2, axis=0):
        pass

class BodyweightSquat(Exercise):
    def __init__(self, side=Side.RIGHT):
        self.side = side

    def keypoints(self):
        if self.side == Side.RIGHT:
            return [Keypoints.RHIP, Keypoints.RKNEE, Keypoints.RANKLE, Keypoints.NECK]
        return [Keypoints.LHIP, Keypoints.LKNEE, Keypoints.LANKLE, Keypoints.NECK]

    def compute_center(self, pose):
        return pose[0, :] # hip
    
    def center(self, seq, width=1, height=1):
        assert seq.shape[0] == 4
        global_center = np.array([width / 2, height / 2])
        global_center[1] -= 0.1
        center = self.compute_center(seq[:, :, 0])
        diff = global_center - center
        for i in range(2):
            seq[:, i, :] += diff[i]

    def normalize(self, seq1, seq2, axis=0):
        pose1 = seq1[:, :, 0]
        pose2 = seq2[:, :, 0]

        if axis == 0:
            pass
        else:
            height1 = pose1[3, 1] - pose1[2, 1]
            height2 = pose2[3, 1] - pose2[2, 1]
            r = height1 / height2
        
        seq2[:, axis, :] *= r

    def detect_reps(self, seq, drop_percent=0.80):
        neck = Keypoints.NECK.value
        y = 1

        reps = []
        rep = []
        s = 1 - seq

        initial = s[neck, y, 0] # assumes standing start pos
        # print(initial)
        in_rep = False
        for t in range(1, seq.shape[-1]):
            # print(s[neck, y, t])
            if in_rep:
                # print("in")
                if s[neck, y, t] >= drop_percent * initial:
                    # print("end")
                    rep.append(t)
                    reps.append(rep)
                    rep = []
                    in_rep = False
            elif s[neck, y, t] < drop_percent * initial:
                # print("start")
                rep.append(t)
                in_rep = True
        print(reps)
        return np.array(reps, dtype=np.uint32)

    def detect_side(self, seq):
        hip = Keypoints.RHIP.value
        knee = Keypoints.RKNEE.value

        initial = seq[:, :, 0]
        b = (initial[hip, 1] - initial[knee, 1]) / (initial[hip, 0] - initial[knee, 0])
        if b > 0:
            self.side = Side.RIGHT
        else:
            self.side = Side.LEFT

def load(exercise_type_str):
    exercise_type = ExerciseType(exercise_type_str)
    if exercise_type == ExerciseType.BS:
        return BodyweightSquat()
