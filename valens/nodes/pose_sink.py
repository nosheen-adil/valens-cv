import valens as va
from valens import constants
from valens.node import Node
from valens.stream import InputStream, gen_addr_ipc
import valens.pose

import cv2
import h5py
import json
import numpy as np
import trt_pose.coco

class PoseSink(Node):
    def __init__(self, sequences_dir=constants.DATA_DIR + '/sequences', human_pose_path=constants.POSE_JSON):
        super().__init__("PoseSink")
        self.input_streams["pose"] = InputStream(gen_addr_ipc('pose'), identity=b'0')
        with open(human_pose_path, 'r') as f:
            human_pose = json.load(f)
            total_keypoints = len(human_pose["keypoints"])

        self.total_frames = None
        self.filename = None
        self.sequences_dir = sequences_dir
        self.frames = 0
        self.data = None
        
    def reset(self):
        self.total_frames = None
        self.filename = None
        self.data = None
        self.frames = 0

    def configure(self, request):
        def total_frames(url):
            c = cv2.VideoCapture(url)
            return int(c.get(cv2.CAP_PROP_FRAME_COUNT))

        self.total_frames = total_frames(request['capture'])
        self.data = np.empty((len(va.pose.Keypoints), 2, self.total_frames))
        self.data[:] = np.nan

        self.filename = self.sequences_dir + '/' + request['name'] + '.h5'

    def process(self):
        print(self.name + ':', self.frames)
        pose, _ = self.input_streams["pose"].recv() # total_keypoints x 2
        if pose is None:
            self.save()
            return True

        self.data[:, :, self.frames] = pose
        self.frames += 1

    def save(self):
        with h5py.File(self.filename, 'w') as f:
            f["pose"] = self.data
        print(self.name + ": saved keypoints to", self.filename)
