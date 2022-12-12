from valens import constants
from valens import exercise
from valens import pose
from valens import feedback
from valens.node import Node
from valens.stream import InputStream

import valens as va

from abc import abstractmethod
import cv2
import json
import trt_pose.coco
import numpy as np

def get_capture_dims(name):
    c = cv2.VideoCapture(name)
    return int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))

class VideoSink(Node):
    def __init__(self, node_name='VideoSink'):
        super().__init__(node_name, topics=['feedback_frame'])

        self.right_topology = None
        self.left_topology = None
        self.set_id = None
        
        # self.input_streams['feedback_frame'] = InputStream(va.stream.gen_addr_tcp(7000, bind=False), identity=b'0')

    def reset(self):
        self.right_topology = None
        self.left_topology = None
        self.set_id = None

    def configure(self, request):
        exercise = va.exercise.load(request['exercise'])
        self.right_topology = exercise.topology(side=va.exercise.Side.RIGHT)
        self.left_topology = exercise.topology(side=va.exercise.Side.LEFT)
        self.set_id = request['set_id']

    def process(self):
        # print('writing', self.iterations)
        # frame, _ = self.input_streams['feedback_frame'].recv()
        # if frame is None:
        #     return True
        frame = self.bus.recv('feedback_frame', timeout=100)
        if frame is False:
            finished = self.bus.recv('finished')
            if finished is True:
                print(self.name + ': caught finished, exiting')
                return True
            return
        
        self.write(frame)

    @abstractmethod
    def write(self, frame):
        pass

class LocalDisplaySink(VideoSink):
    def __init__(self):
        super().__init__(node_name='LocalDisplaySink')

    def write(self, frame):
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

class Mp4FileSink(VideoSink):
    def __init__(self, output_dir=constants.DATA_DIR + '/outputs', fps=10):
        super().__init__(node_name='Mp4FileSink')
        self.output_dir = output_dir
        self.fps = fps
        self.filename = None
        self.writer = None
        self.width = None
        self.height = None

    def reset(self):
        print(self.name + ': closing writer to ', self.filename)
        self.writer.release()
        # self.writer = None

        self.filename = None
        self.writer = None
        self.width = None
        self.height = None
        super().reset()

    def configure(self, request):
        super().configure(request)

        self.filename = self.output_dir + '/' + request['name'] + '.mp4'
        self.width, self.height = get_capture_dims(request['capture'])
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (self.width, self.height))

    def write(self, frame):
        self.writer.write(frame)
