from valens import constants
from valens.node import Node
from valens.stream import OutputStream

import cv2

class VideoSource(Node):
    def __init__(self, frame_address, device_url=0, num_outputs=1, resize=True, max_fps=10):
        super().__init__("VideoSource")
        self.output_streams["frame"] = OutputStream(frame_address, num_outputs=num_outputs)
        
        self.capture = None
        self.device_url = device_url
        self.resize = resize

        self.set_max_fps(max_fps)
        self.diff_frames = 0
        self.t = 0
    
    def prepare(self):
        self.capture = cv2.VideoCapture(self.device_url)
        original_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.diff_frames = round(original_fps / self.max_fps)
        print(self.name, 'diff frames:', self.diff_frames, original_fps)

    def process(self):
        ret, frame = self.capture.read()
        self.t += 1
        if not ret:
            self.capture.release()
            self.stop()
            return

        if self.resize:
            frame = cv2.resize(frame, dsize=(constants.POSE_MODEL_WIDTH, constants.POSE_MODEL_HEIGHT))
        
        self.output_streams["frame"].send(frame)
        
        for _ in range(self.diff_frames):
            self.t += 1
            _, _, = self.capture.read()
            