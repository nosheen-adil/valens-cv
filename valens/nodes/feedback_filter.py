from valens import constants
from valens.structures.node import Node
from valens.structures.stream import InputStream, OutputStream, gen_addr_ipc
# from valens.structures import exercise

import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
import json

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

class Exercise(ABC):
    def __init__(self):
        self.keypoints = None # change to all keypoints by default
        self.mask = None
        self.topology = None
        self.in_rep = False
        self.initial = None
        self.rep = []
        self.x = 0
        self.y = 1
        self.t = 0

    def initialize(self, pose):
        self.calc_side(pose)
        self.calc_keypoint_mask()
        self.calc_topology()
        self.initial = pose[self.mask]

    def calc_side(self, pose):
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
        if self.initial is None:
            self.initialize(pose)
            self.t += 1
            return

        filtered = pose[self.mask]
        if self.in_rep:
            self.rep.append(filtered)
            if self.finished(filtered):
                print('finished!', self.t)
                self.eval()
                self.in_rep = False
                self.rep = []
            else:
                self.recommend()
        elif self.started(filtered):
            print('started rep!', self.t)
            self.rep.append(filtered)
            self.in_rep = True

        self.t += 1

    def eval(self):
        pass
    
    def recommend(self):
        pass

    @abstractmethod
    def started(self, pose):
        pass

    @abstractmethod
    def finished(self, pose):
        pass

class BodyweightSquat(Exercise):
    def __init__(self, drop_percent=0.8):
        super().__init__()
        self.drop_percent = drop_percent

        self.hip = 0
        self.knee = 1
        self.ankle = 2
        self.neck = 3

    def calc_side(self, pose):
        hip = Keypoints.RHIP.value
        knee = Keypoints.RKNEE.value

        b = (pose[hip, 1] - pose[knee, 1]) / (pose[hip, 0] - pose[knee, 0])
        if b > 0:
            print("detected right side")
            self.keypoints = [Keypoints.RHIP, Keypoints.RKNEE, Keypoints.RANKLE, Keypoints.NECK]
        else:
            print("detected left side")
            self.keypoints = [Keypoints.LHIP, Keypoints.LKNEE, Keypoints.LANKLE, Keypoints.NECK]
    
    def started(self, pose):
        return pose[self.neck, self.y] > (1 - self.drop_percent * (1 - self.initial[self.neck, self.y]))

    def finished(self, pose):
        return pose[self.neck, self.y] <= (1 - self.drop_percent * (1 - self.initial[self.neck, self.y]))

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
            self.output_streams['pose'].send(feedback)
