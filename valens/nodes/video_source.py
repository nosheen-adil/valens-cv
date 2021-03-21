from valens import constants
from valens.node import Node
from valens.stream import OutputStream, gen_set_id, gen_sync_metadata, gen_addr_ipc
from valens.exercise import ExerciseType

import cv2

class VideoSource(Node):
    def __init__(self, frame_address=gen_addr_ipc("frame"), is_live=False, num_outputs=1, resize=True, max_fps=10):
        super().__init__("VideoSource")
        
        self.output_streams["frame"] = OutputStream(frame_address, num_outputs=num_outputs)
        self.is_live = is_live
        self.resize = resize
        self.set_max_fps(max_fps)

        self.user_id = None
        self.exercise = None
        self.set_id = None
        self.diff_frames = 0
        self.t = 0
        self.capture = None

    def reset(self):
        self.user_id = None
        self.exercise = None
        self.set_id = None
        self.diff_frames = 0
        self.t = 0
        self.capture = None

    def configure(self, request):
        self.user_id = request['user_id']
        self.exercise = str(ExerciseType(request['exercise']))
        self.set_id = gen_set_id(16)

        self.capture = cv2.VideoCapture(request['capture'])
        original_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.diff_frames = round(original_fps / self.max_fps) - 1
        print(self.name, 'diff frames:', self.diff_frames, original_fps)

    def process(self):
        ret, frame = self.capture.read()
        self.t += 1
        if not ret:
            self.capture.release()
            # self.stop()
            self.bus.send("finished")
            return

        if self.resize:
            frame = cv2.resize(frame, dsize=(constants.POSE_MODEL_WIDTH, constants.POSE_MODEL_HEIGHT))
        
        sync = gen_sync_metadata(self.user_id, self.exercise, self.set_id)
        self.output_streams["frame"].send(frame, sync)
        
        if not self.is_live:
            for _ in range(self.diff_frames):
                self.t += 1
                _, _, = self.capture.read()
            