from valens import constants
from valens import pose
from valens.node import Node
from valens.stream import InputStream

import cv2
import h5py
import json
import numpy as np
import trt_pose.coco

class PoseSink(Node):
    def __init__(self, pose_address, total_frames, filename, human_pose_path=constants.POSE_JSON):
        super().__init__("PoseSink")
        self.input_streams["pose"] = InputStream(pose_address)
        with open(human_pose_path, 'r') as f:
            human_pose = json.load(f)
            total_keypoints = len(human_pose["keypoints"])

        self.total_frames = total_frames
        self.frames = 0
        self.filename = filename

        self.data = np.empty((total_keypoints, 2, self.total_frames))
        self.data[:] = np.nan

    def process(self):
        p = self.input_streams["pose"].recv() # total_keypoints x 2
        if p is None:
            self.save()
            self.stop()
            return

        self.data[:, :, self.frames] = p
        self.frames += 1

    def save(self):
        print(self.name + ": saving keypoints")
        with h5py.File(self.filename, 'w') as f:
            f["pose"] = self.data
