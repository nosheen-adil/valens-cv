from valens import constants
from valens.structures.node import Node
from valens.structures.stream import OutputStream

import cv2

class VideoSource(Node):
    def __init__(self, frame_address, device_url=0, num_outputs=1, resize=True):
        super().__init__("VideoSource")
        self.output_streams["frame"] = OutputStream(frame_address, num_outputs=num_outputs)
        
        self.capture = None
        self.device_url = device_url
        self.resize = resize
    
    def prepare(self):
        self.capture = cv2.VideoCapture(self.device_url)

    def process(self):
        ret, frame = self.capture.read()
        if not ret:
            self.stop()
            return

        if self.resize:
            frame = cv2.resize(frame, dsize=(constants.POSE_MODEL_WIDTH, constants.POSE_MODEL_HEIGHT))
        
        self.output_streams["frame"].send(frame)
