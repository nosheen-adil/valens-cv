from valens import constants
from valens import structures as va
from valens.structures.pose import Keypoints
import valens.structures.sequence

from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import math

class ExerciseType(Enum):
    BS = "BS"

class Exercise(ABC):
    def __init__(self, exercise_type, keypoints, topology, window_size=3, damping_factor=0.5, diff_limit=0.1):
        self.keypoints = keypoints
        self.keypoint_mask = va.pose.keypoint_mask(keypoints)
        self.topology = topology

        self.rep = [[] for _ in range(len(topology))]
        self.prev_rep = None
        self.prev_label = None
        self.prev_match = None
        self.in_rep = False
        self.zero_pos = None
        self.window = []
        self.window_size = window_size
        self.damping_factor = damping_factor
        self.diff_limit = diff_limit

        self.classifier = va.sequence.DtwKnn()
        X_train, y_train = va.sequence.load_features(va.sequence.post_processed_filenames(exercise_type.value))
        self.classifier.fit(X_train, y_train)

    @abstractmethod
    def started(self, angles):
        pass

    @abstractmethod
    def finished(self, angles):
        pass

    @abstractmethod
    def joint_angles(self, angles):
        pass

    def fit(self, pose):
        filtered = pose[self.keypoint_mask]
        if np.isnan(filtered).any():
            if self.zero_pos is None:
                return
            else:
                assert not np.isnan(self.window[-1].any())
                mask = np.isnan(filtered)
                filtered[mask] = self.window[-1][mask]
        
        self.window.append(filtered.copy())
        if len(self.window) == self.window_size + 1:
            self.window.pop(0)
            for k in range(len(self.keypoints)):
                x = np.array([self.window[t][k, :] for t in range(len(self.window))])
                filtered[k, :] = np.mean(x, axis=0)

        angles = va.pose.angles(filtered, self.topology)
        if self.zero_pos is None:
            self.zero_pos = angles

        if self.in_rep:
            for k in range(angles.shape[0]):
                self.rep[k].append(angles[k])

            if self.finished(angles):
                self.eval()
                self.in_rep = False
                self.rep = [[] for _ in range(len(self.rep))]
            
        elif self.started(angles):
            for k in range(angles.shape[0]):
                self.rep[k].append(angles[k])
            self.in_rep = True

    def eval(self):
        rep = np.array(self.rep)
        va.sequence.filter_noise(rep)
        label, match = self.classifier.predict(rep)
        self.prev_rep = rep
        self.prev_label = label
        self.prev_match = match

    def predict(self):
        if self.zero_pos is None:
            return

        if self.in_rep and self.prev_rep is not None:
            return va.feedback.to_json(self.window[-1], corrected_pose=self.project(), keypoints=self.keypoints)
            # return np.array(self.window[-1], self.project())
        
        return va.feedback.to_json(self.window[-1], keypoints=self.keypoints)
        # return self.window[-1]

    def project(self):
        num_angles = len(self.rep)
        user_angles = np.array([self.rep[feature][-1] for feature in range(num_angles)])
        ref_angles = np.array([self.prev_match[feature, -1] for feature in range(num_angles)])
        diff = user_angles - ref_angles
        diff[diff < self.diff_limit] = 0.0
        user_angles += diff * self.damping_factor
        joint_angles = self.joint_angles(user_angles)
        ref_pose = self.window[-1]
        corrected_pose = va.pose.transform(ref_pose, joint_angles, self.topology)
        return corrected_pose

class BodyweightSquat(Exercise):
    def __init__(self, drop_percent=0.75):
        super().__init__(
            exercise_type=ExerciseType.BS,
            keypoints=[Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK],
            topology=[[0, 1], [1, 2], [2, 3]]
        )
        self.drop_percent = drop_percent

    def started(self, angles):
        return (np.pi - angles[0]) < self.drop_percent * (np.pi - self.zero_pos[0])

    def finished(self, angles):
        return (np.pi - angles[0]) >= self.drop_percent * (np.pi - self.zero_pos[0])

    def joint_angles(self, angles):
        return np.array([angles[0] - np.pi, -angles[1], angles[2], 0])

def load(exercise_type_str):
    exercise_type = ExerciseType(exercise_type_str)
    if exercise_type == ExerciseType.BS:
        return BodyweightSquat()

# class Side(Enum):
#     RIGHT = 0
#     LEFT = 1

# class Exercise(ABC):
#     @abstractmethod
#     def keypoints(self):
#         pass

#     def keypoint_mask(self):
#         keypoints = self.keypoints()
#         mask = np.zeros((len(Keypoints),), dtype=np.bool)
#         for keypoint in keypoints:
#             mask[keypoint.value] = 1
#         return mask

#     def topology(self, human_pose_path=constants.POSE_JSON):
#         keypoints = self.keypoints()
#         with open(human_pose_path, 'r') as f:
#             human_pose = json.load(f)

#         skeleton = human_pose['skeleton']
#         K = len(skeleton)
#         topology = []
#         for k in range(K):
#             a = Keypoints(skeleton[k][0] - 1)
#             b = Keypoints(skeleton[k][1] - 1)
#             if a in keypoints and b in keypoints:
#                 topology.append([keypoints.index(a), keypoints.index(b)])
#         return topology

#     @abstractmethod
#     def compute_center(self, pose):
#         pass
    
#     @abstractmethod
#     def center(self, seq, width=1, height=1):
#         pass

#     @abstractmethod
#     def normalize(self, seq1, seq2, axis=0):
#         pass

# class BodyweightSquat(Exercise):
#     def __init__(self, side=Side.RIGHT):
#         self.side = side

#     def keypoints(self):
#         if self.side == Side.RIGHT:
#             return [Keypoints.RHIP, Keypoints.RKNEE, Keypoints.RANKLE, Keypoints.NECK]
#         return [Keypoints.LHIP, Keypoints.LKNEE, Keypoints.LANKLE, Keypoints.NECK]

#     def compute_center(self, pose):
#         return pose[0, :] # hip
    
#     def center(self, seq, width=1, height=1):
#         assert seq.shape[0] == 4
#         global_center = np.array([width / 2, height / 2])
#         global_center[1] -= 0.1
#         center = self.compute_center(seq[:, :, 0])
#         diff = global_center - center
#         for i in range(2):
#             seq[:, i, :] += diff[i]

#     def normalize(self, seq1, seq2, axis=0):
#         pose1 = seq1[:, :, 0]
#         pose2 = seq2[:, :, 0]

#         if axis == 0:
#             pass
#         else:
#             height1 = pose1[3, 1] - pose1[2, 1]
#             height2 = pose2[3, 1] - pose2[2, 1]
#             r = height1 / height2
        
#         seq2[:, axis, :] *= r

#     def detect_reps(self, seq, drop_percent=0.80):
#         neck = Keypoints.NECK.value
#         y = 1

#         reps = []
#         rep = []
#         s = 1 - seq

#         initial = s[neck, y, 0] # assumes standing start pos
#         # print(initial)
#         in_rep = False
#         for t in range(1, seq.shape[-1]):
#             # print(s[neck, y, t])
#             if in_rep:
#                 # print("in")
#                 if s[neck, y, t] >= drop_percent * initial:
#                     # print("end")
#                     rep.append(t)
#                     reps.append(rep)
#                     rep = []
#                     in_rep = False
#             elif s[neck, y, t] < drop_percent * initial:
#                 # print("start")
#                 rep.append(t)
#                 in_rep = True
#         print(reps)
#         return np.array(reps, dtype=np.uint32)

#     def detect_side(self, seq):
#         hip = Keypoints.RHIP.value
#         knee = Keypoints.RKNEE.value

#         initial = seq[:, :, 0]
#         b = (initial[hip, 1] - initial[knee, 1]) / (initial[hip, 0] - initial[knee, 0])
#         if b > 0:
#             self.side = Side.RIGHT
#         else:
#             self.side = Side.LEFT
