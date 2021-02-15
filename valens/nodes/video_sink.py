import cv2
import trt_pose.coco
import json

from valens.structures.node import Node
from valens.structures.stream import InputStream
from valens.structures import pose

class VideoSink(Node):
    def __init__(self, frame_address, pose_address=None, pose_path=None):
        super().__init__("VideoSink")
        self.input_streams["frame"] = InputStream(frame_address)
        if pose_address is not None:
            self.input_streams["pose"] = InputStream(pose_address)
            self.topology = pose.get_topology(pose_path)

    def process(self):
        frame = self.input_streams["frame"].recv()
        if "pose" in self.input_streams:
            k = self.input_streams["pose"].recv()
            pose.scale(k, frame.shape[1], frame.shape[0])
            pose.draw_on_image(self.topology, frame, k)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
