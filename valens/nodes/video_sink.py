from valens import constants
from valens.structures import pose
from valens.structures.node import Node
from valens.structures.stream import InputStream

import cv2
import json
import trt_pose.coco

class VideoSink(Node):
    def __init__(self, frame_address, pose_address=None, pose_path=constants.POSE_JSON):
        super().__init__("VideoSink")
        self.input_streams["frame"] = InputStream(frame_address)
        if pose_address is not None:
            self.input_streams["pose"] = InputStream(pose_address)
            self.topology = pose.get_topology(pose_path)

    def process(self):
        frame = self.input_streams["frame"].recv()
        if frame is None:
            p = self.input_streams["pose"].recv()
            assert(p is None)
            self.stop()
            return

        if "pose" in self.input_streams:
            p = self.input_streams["pose"].recv()
            if p is None:
                self.stop()
                return

            pose.draw_on_image(p, frame, self.topology)
            
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
