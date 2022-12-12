from valens import constants
from valens.node import Node
from valens.stream import OutputStream, gen_set_id, gen_sync_metadata, gen_addr_ipc
from valens.exercise import ExerciseType

import cv2
import time

class VideoSource(Node):
    def __init__(self, is_live=False, max_fps=10, num_outputs=1):
        super().__init__("VideoSource")
        
        self.output_streams["frame"] = OutputStream(gen_addr_ipc('frame'), identities=[b'0'] if num_outputs == 1 else [b'0', b'1'])
        self.is_live = is_live
        self.set_max_fps(max_fps)
        
        self.capture = None

        self.user_id = None
        self.exercise = None
        self.set_id = None
        self.diff_frames = 0
        self.t = 0
        

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
        self.set_id = request['set_id']

        self.capture = cv2.VideoCapture(request['capture'])
        original_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.diff_frames = round(original_fps / self.max_fps) - 1
        print(self.name, 'diff frames:', self.diff_frames, original_fps)

    def process(self):
        finished = self.bus.recv('finished')
        if finished is True:
            print('VideoSource: caught finished, forwarding sentinel')
            self.output_streams['frame'].send() # forward sentinel
            return True
        
        ret, frame = self.capture.read()
        self.t += 1
        if not ret:
            self.capture.release()
            print('VideoSource: finished video, forwarding sentinel')
            
            self.bus.send("finished")
            self.output_streams['frame'].send() # forward sentinel
            return True
        
        # print(self.name + ': sending')
        sync = gen_sync_metadata(self.user_id, self.exercise, self.set_id)
        sync['id'] = self.iterations
        self.output_streams['frame'].send(frame, sync)
        # print(self.name + ': sent')
        if not self.is_live:
            for _ in range(self.diff_frames):
                self.t += 1
                _, _, = self.capture.read()
            