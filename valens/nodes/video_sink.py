import cv2
import trt_pose.coco
import json

from valens.structures.node import Node
from valens.structures.stream import InputStream
from valens.structures import keypoints

class VideoSink(Node):
    def __init__(self, frame_address, keypoints_address=None, pose_path=None):
        super().__init__("VideoSink")
        self.input_streams["frame"] = InputStream(frame_address)
        if keypoints_address is not None:
            self.input_streams["keypoints"] = InputStream(keypoints_address)
            self.topology = keypoints.get_topology(pose_path)

    def process(self):
        frame = self.input_streams["frame"].recv()
        if "keypoints" in self.input_streams:
            k = self.input_streams["keypoints"].recv()
            keypoints.scale(k, frame.shape[1], frame.shape[0])
            keypoints.draw_on_image(self.topology, frame, k)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
