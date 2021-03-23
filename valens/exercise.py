import valens as va
from valens.pose import Keypoints

from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import math

class ExerciseType(Enum):
    BS = "BS"
    BC = "BC"
    PU = "PU"

    def __str__(self):
        if self is ExerciseType.BS:
            return "bodyweight-squat"
        elif self is ExerciseType.BC:
            return "bicep-curl"
        elif self is ExerciseType.PU:
            return "push-up"
        assert False

class Side(Enum):
    RIGHT = 0
    LEFT = 1

class Exercise(ABC):
    def __init__(self, exercise_type, right_keypoints, left_keypoints, topology, window_size=3):
        self.type = exercise_type
        self.left_keypoints = left_keypoints
        self.right_keypoints = right_keypoints
        self.id_topology = topology

        self.side = None
        self._max_frames_side_detected = 1
        self._curr_frames_side_detected = 0

        self.rep = []
        self.in_rep = False
        self.window = []
        self.window_size = window_size
        self.pose = None

        self.t = 0
        self.reps = 0

    @abstractmethod
    def started(self, angles):
        pass

    @abstractmethod
    def finished(self, angles):
        pass

    @abstractmethod
    def detect_side(self):
        pass

    @abstractmethod
    def angles(self, filtered):
        pass

    @abstractmethod
    def correct(self):
        pass

    @abstractmethod
    def transform(self, angles):
        pass

    def keypoints(self, side=None):
        if side is Side.LEFT or self.side is Side.LEFT:
            return self.left_keypoints
        else:
            return self.right_keypoints

    def topology(self, side=None):
        if side is Side.LEFT or self.side is Side.LEFT:
            return va.feedback.topology(self.id_topology, self.left_keypoints)
        else:
            return va.feedback.topology(self.id_topology, self.right_keypoints)

    def fit(self, pose):
        if self._curr_frames_side_detected < self._max_frames_side_detected:
            side = self.detect_side(pose)
            if side is not None:
                if side is not self.side:
                    print('resetting side detection')
                    self._curr_frames_side_detected = 0
                else:
                    print('same side detected')
                    self._curr_frames_side_detected += 1
            self.side = side

        filtered = va.pose.filter_keypoints(pose, self.keypoints())
        if np.isnan(filtered).any():
            if len(self.window) > 0:
                assert not np.isnan(self.window[-1].any())
                mask = np.isnan(filtered)
                filtered[mask] = self.window[-1][mask]
        
        self.window.append(filtered.copy())
        if len(self.window) == self.window_size + 1:
            self.window.pop(0)
            for k in range(len(self.keypoints())):
                x = np.array([self.window[t][k, :] for t in range(self.window_size)])
                filtered[k, :] = np.mean(x, axis=0)

        self.pose = filtered

        angles = self.angles(filtered)
        if self.in_rep:
            for k in range(angles.shape[0]):
                self.rep[k].append(angles[k])

            if self.finished(angles):
                print('finished', self.t)
                self.in_rep = False
                self.rep = []
                self.reps += 1
            
        elif self.started(angles):
            print('started:', self.t)
            for k in range(angles.shape[0]):
                self.rep.append([angles[k]])
            self.in_rep = True

        self.t += 1

    def predict(self):
        if len(self.window) == 0:
            return

        if self.in_rep:
            labels, corrected_angles = self.correct()
            corrected_pose = self.transform(corrected_angles)
            return va.feedback.to_json(
                user_pose=self.pose,
                labels=labels,
                corrected_pose=corrected_pose,
                keypoints=self.keypoints())

        return va.feedback.to_json(
            user_pose=self.pose,
            labels=[va.feedback.KeypointResult.UNDEFINED for _ in range(self.pose.shape[0])],
            keypoints=self.keypoints())

class BodyweightSquat(Exercise):
    def __init__(self):
        super().__init__(
            exercise_type=ExerciseType.BS,
            right_keypoints=[Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK],
            left_keypoints=[Keypoints.LANKLE, Keypoints.LKNEE, Keypoints.LHIP, Keypoints.NECK],
            topology=[[0, 1], [1, 2], [2, 3]]
        )
        self._knee_to_neck_limit = math.radians(120)
        self._knee_to_neck_prev_diff_limit = 1.0
        self._space_to_knee_min_good = math.radians(160)
        self._knee_to_neck_min_good = math.radians(90)
        self._hip_to_ankle_min_good = math.radians(90)
        self._max_percent_diff = 0.15

    def started(self, angles):
        return (np.pi - angles[2]) < self._knee_to_neck_limit

    def finished(self, angles):
        return (np.pi - angles[2]) >= self._knee_to_neck_limit

    def detect_side(self, pose):
        knee = 1
        hip = 2
        filtered = va.pose.filter_keypoints(pose, self.right_keypoints)
        if np.isnan(filtered[knee, :]).any() or np.isnan(filtered[hip, :]).any():
            filtered = va.pose.filter_keypoints(pose, self.left_keypoints)
            if np.isnan(filtered[knee, :]).any() or np.isnan(filtered[hip, :]).any():
                return
        
        b = (filtered[knee, 1] - filtered[hip, 1]) / (filtered[knee, 0] - filtered[hip, 0])
        if b > 0:
            print('right side')
            return Side.RIGHT
        else:
            print('left side')
            return Side.LEFT

    def angles(self, filtered):
        return va.pose.angles(filtered, space_frame=[0, 1])

    def joint_angles(self, angles):
        if self.side is Side.LEFT:
            return np.array([-np.pi + angles[0], angles[1], -angles[2], 0])
        else:
            return np.array([np.pi - angles[0], -angles[1], angles[2], 0])

    def correct(self):
        assert len(self.rep) == 3
        user_angles = np.array([self.rep[feature][-1] for feature in range(len(self.rep))])
        labels = [va.feedback.KeypointResult.GOOD for _ in range(self.pose.shape[0])]
        corrected_angles = user_angles[:]
        if len(self.rep[0]) > 1:
            dt = 1.0 / 10.0
            diff = ((np.pi - user_angles[2]) - (np.pi - self.rep[2][-2])) / dt
            if abs(diff) < self._knee_to_neck_prev_diff_limit:
                x = (((np.pi - user_angles[2]) - self._knee_to_neck_min_good) / self._knee_to_neck_min_good)
                if x < 0 and abs(x) > self._max_percent_diff:
                    print('bad!')
                    labels = [va.feedback.KeypointResult.BAD for _ in range(self.pose.shape[0])]
                    corrected_angles = [self._space_to_knee_min_good, self._hip_to_ankle_min_good, self._knee_to_neck_min_good]

        return labels, corrected_angles

    def transform(self, angles):
        user_pose = self.pose
        num_joints = user_pose.shape[0]
        L = [0.0]
        for i in range(1, num_joints):
            joint1 = i - 1
            joint2 = i

            dist = np.linalg.norm(user_pose[joint2, :] - user_pose[joint1, :])
            L.append(dist + L[i - 1])

        zero_pose = np.empty((num_joints, 2))
        for i in range(num_joints):
            zero_pose[i, :] = [user_pose[0, 0], user_pose[0, 1] - L[i]]

        Slist = np.empty((6, num_joints))
        for i in range(num_joints):
            Slist[:, i] = [0, 0, 1, -L[i], 0, 0]

        joint_angles = self.joint_angles(angles)
        corrected_pose = va.pose.transform(zero_pose, Slist, joint_angles)
        return corrected_pose

class BicepCurl(Exercise):
    def __init__(self):
        super().__init__(
            exercise_type=ExerciseType.BC,
            right_keypoints=[Keypoints.NECK, Keypoints.RHIP, Keypoints.RSHOULDER, Keypoints.RELBOW, Keypoints.RWRIST],
            left_keypoints=[Keypoints.NECK, Keypoints.LHIP, Keypoints.LSHOULDER, Keypoints.LELBOW, Keypoints.LWRIST],
            topology=[[0, 2], [1, 2], [2, 3], [3, 4]]
        )
        self._angle_mask = np.array([False, True, True, True, True], dtype=np.bool)
        self._shoulder_wrist_limit = math.radians(120)
        self._shoulder_wrist_diff_limit = 0.3
        self._shoulder_wrist_min_good = math.radians(30)
        self._hip_to_elbow_min_good = math.radians(6)
        self._max_percent_diff = 0.1

    def started(self, angles):
        return (np.pi - angles[1]) < self._shoulder_wrist_limit

    def finished(self, angles):
        return (np.pi - angles[1]) >= self._shoulder_wrist_limit

    def detect_side(self, pose):
        shoulder = 2
        wrist = 4
        filtered = va.pose.filter_keypoints(pose, self.right_keypoints)
        if np.isnan(filtered[shoulder, :]).any() or np.isnan(filtered[wrist, :]).any():
            filtered = va.pose.filter_keypoints(pose, self.left_keypoints)
            if np.isnan(filtered[shoulder, :]).any() or np.isnan(filtered[wrist, :]).any():
                return
        
        b = (filtered[shoulder, 1] - filtered[wrist, 1]) / (filtered[shoulder, 0] - filtered[wrist, 0])
        if b < 0:
            print('left side')
            return Side.LEFT
        else:
            print('right side')
            return Side.RIGHT

    def angles(self, filtered):
        return va.pose.angles(filtered[self._angle_mask], space_frame=None)

    def joint_angles(self, angles):
        if self.side is Side.LEFT:
            return np.array([np.pi - angles[0], angles[1], 0])
        else:
            return np.array([angles[0], angles[1], 0])

    def correct(self):
        assert len(self.rep) == 2
        user_angles = np.array([self.rep[feature][-1] for feature in range(len(self.rep))])
        labels = [va.feedback.KeypointResult.GOOD for _ in range(self.pose.shape[0])]
        corrected_angles = user_angles[:]
        if len(self.rep[0]) > 1:
            dt = 1.0 / 10.0
            diff = ((np.pi - user_angles[1]) - (np.pi - self.rep[1][-2])) / dt
            if abs(diff) < self._shoulder_wrist_diff_limit:
                x = (((np.pi - user_angles[0]) - self._hip_to_elbow_min_good) / self._hip_to_elbow_min_good)
                if x > 0 and x > self._max_percent_diff:
                    print('bad!')
                    labels = [va.feedback.KeypointResult.BAD for _ in range(self.pose.shape[0])]
                    corrected_angles = [np.pi - self._hip_to_elbow_min_good, np.pi - self._shoulder_wrist_min_good]
        return labels, corrected_angles
            
    def transform(self, angles):
        user_pose = self.pose[2:, :]
        num_joints = user_pose.shape[0]
        L = [0.0]
        for i in range(1, num_joints):
            joint1 = i - 1
            joint2 = i

            dist = np.linalg.norm(user_pose[joint2, :] - user_pose[joint1, :])
            L.append(dist + L[i - 1])

        zero_pose = np.empty((num_joints, 2))
        for i in range(num_joints):
            zero_pose[i, :] = [user_pose[0, 0], user_pose[0, 1] + L[i]]
        
        Slist = np.empty((6, num_joints))
        for i in range(num_joints):
            Slist[:, i] = [0, 0, 1, L[i], 0, 0]

        joint_angles = self.joint_angles(angles)
        _corrected_pose = va.pose.transform(zero_pose, Slist, joint_angles)
        corrected_pose = self.pose.copy()
        corrected_pose[2:] = _corrected_pose

        return corrected_pose

class PushUp(Exercise):
    def __init__(self):
        super().__init__(
            exercise_type=ExerciseType.PU,
            right_keypoints=[Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK, Keypoints.RSHOULDER, Keypoints.RELBOW, Keypoints.RWRIST],
            left_keypoints=[Keypoints.LANKLE, Keypoints.LKNEE, Keypoints.LHIP, Keypoints.NECK, Keypoints.LSHOULDER, Keypoints.LELBOW, Keypoints.LWRIST],
            topology=[[0, 1], [1, 2], [2, 4], [4, 3], [4, 5], [5, 6]]
        )
        self._space_neck_limit = math.radians(20)
        
        self._shoulder_wrist_start_limit = math.radians(150)
        self._shoulder_wrist_frames_from_start = 5
        self._shoulder_wrist_frames_from_start_curr = 0
        self._shoulder_wrist_correct = True

        self._initial_arm_link_lengths = None
        self._shoulder_elbow_arm_ratio = 0.25

        self._shoulder_wrist_good = math.radians(90)
        self._space_elbow_good = math.radians(45)
    
    def started(self, angles):
        if self._initial_arm_link_lengths is None:
            initial_pose = self.pose.copy()
            if (not np.isnan(initial_pose[4, :]).any() and not np.isnan(initial_pose[6, :]).any()):
                L = np.linalg.norm(initial_pose[4, :] - initial_pose[6, :])
                shoulder_elbow = self._shoulder_elbow_arm_ratio * L
                elbow_wrist = self._shoulder_elbow_arm_ratio * L
                self._initial_arm_link_lengths = (shoulder_elbow, elbow_wrist)
                print('got initial!', shoulder_elbow, elbow_wrist)
        self._shoulder_wrist_correct = True
        self._shoulder_wrist_frames_from_start_curr = 0
        return angles[0] < self._space_neck_limit

    def finished(self, angles):
        k = 2
        if self._shoulder_wrist_frames_from_start_curr < self._shoulder_wrist_frames_from_start:
            if self._shoulder_wrist_correct is True and (np.pi - angles[k]) > self._shoulder_wrist_start_limit:
                self._shoulder_wrist_correct = False
                print('bad')

        self._shoulder_wrist_frames_from_start_curr += 1
        return angles[0] >= self._space_neck_limit

    def detect_side(self, pose):
        hip = 2
        shoulder = 4
        filtered = va.pose.filter_keypoints(pose, self.right_keypoints)
        if np.isnan(filtered[hip, :]).any() or np.isnan(filtered[shoulder, :]).any():
            filtered = va.pose.filter_keypoints(pose, self.left_keypoints)
            if np.isnan(filtered[hip, :]).any() or np.isnan(filtered[shoulder, :]).any():
                return
        
        b = (filtered[hip, 1] - filtered[shoulder, 1]) / (filtered[hip, 0] - filtered[shoulder, 0])
        if b < 0:
            print('right side')
            return Side.RIGHT
        else:
            print('left side')
            return Side.LEFT

    def angles(self, filtered):
        space_neck_pose = np.vstack((filtered[1, :], filtered[4, :]))
        space_neck_angle = va.pose.angles(space_neck_pose, space_frame=[1, 0] if self.side is Side.RIGHT else [-1, 0])
        shoulder_wrist_pose = filtered[4:, :]
        shoulder_wrist_angle = va.pose.angles(shoulder_wrist_pose, space_frame=[0, 1])
        return np.array([space_neck_angle[0], shoulder_wrist_angle[0], shoulder_wrist_angle[1]])

    def arm_joint_angles(self, arm_angles):
        if self.side is Side.RIGHT:
            return np.array([arm_angles[0], -arm_angles[1], 0.0])
        else:
            return np.array([-arm_angles[0], arm_angles[1]  , 0.0])

    def correct(self):
        labels = [va.feedback.KeypointResult.GOOD for _ in range(self.pose.shape[0])]
        num_angles = len(self.rep)
        user_angles = np.array([self.rep[feature][-1] for feature in range(num_angles)])

        if not self._shoulder_wrist_correct:
            labels = [va.feedback.KeypointResult.BAD for _ in range(self.pose.shape[0])]
            user_angles[1] = self._space_elbow_good
            user_angles[2] = self._shoulder_wrist_good

        return labels, user_angles

    def transform(self, angles):
        arm_pose = self.pose[4:, :].copy()
        arm_num_joints = arm_pose.shape[0]
        if self._shoulder_wrist_correct:
            L = [0.0]
            for i in range(1, arm_num_joints):
                joint1 = i - 1
                joint2 = i

                dist = np.linalg.norm(arm_pose[joint2, :] - arm_pose[joint1, :])
                L.append(dist + L[i - 1])
        else:
            L = [0.0, self._initial_arm_link_lengths[0], self._initial_arm_link_lengths[0] + self._initial_arm_link_lengths[1]]

        zero_pose = np.empty((arm_num_joints, 2))
        for i in range(arm_num_joints):
            zero_pose[i, :] = [arm_pose[0, 0], arm_pose[0, 1] + L[i]]
        
        Slist = np.empty((6, arm_num_joints))
        for i in range(arm_num_joints):
            Slist[:, i] = [0, 0, 1, L[i], 0, 0]

        joint_angles = self.arm_joint_angles(angles[1:])
        _corrected_pose = va.pose.transform(zero_pose, Slist, joint_angles)
        corrected_pose = self.pose.copy()
        corrected_pose[4:6, :] = _corrected_pose[0:2, :]

        return corrected_pose

def load(exercise_type_str):
    exercise_type = ExerciseType(exercise_type_str)
    if exercise_type == ExerciseType.BS:
        return BodyweightSquat()
    elif exercise_type == ExerciseType.BC:
        return BicepCurl()
    elif exercise_type == ExerciseType.PU:
        return PushUp()
