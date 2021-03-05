from valens import constants
from valens.structures import exercise
from valens.structures import pose
from valens.structures.node import Node
from valens.structures.stream import InputStream

from abc import abstractmethod
import cv2
import json
import trt_pose.coco
import numpy as np

class VideoSink(Node):
    def __init__(self, frame_address=None, pose_address=None, exercise_type='', width=constants.POSE_MODEL_WIDTH, height=constants.POSE_MODEL_HEIGHT, node_name='VideoSink'):
        super().__init__(node_name)

        self.width = width
        self.height = height
        if frame_address is not None:
            self.input_streams['frame'] = InputStream(frame_address)
        if pose_address is not None:
            self.input_streams['pose'] = InputStream(pose_address)
            e = exercise.load(exercise_type)
            self.keypoint_mask = e.keypoint_mask()
            self.topology = e.topology()

    def process(self):
        if 'frame' in self.input_streams:
            frame = self.input_streams['frame'].recv()
        
            if frame is None:
                p = self.input_streams['pose'].recv()
                assert(p is None)
                self.stop()
                return
        else:
            frame = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        if 'pose' in self.input_streams:
            p = self.input_streams['pose'].recv()
            if p is None:
                self.stop()
                return
            if len(p.shape) == 3:
                colors = [(255, 255, 255), (0, 255, 0)]
                for i in range(p.shape[0]):
                    pose.draw_on_image(p[i, :, :], frame, self.topology, color=colors[i])
            else:
                # pose.draw_on_image(p[self.keypoint_mask], frame, self.topology, color=(255, 255, 255))
                pose.draw_on_image(p, frame, self.topology, color=(255, 255, 255))

        self.write(frame)

    @abstractmethod
    def write(self, frame):
        pass

class LocalDisplaySink(VideoSink):
    def __init__(self, frame_address=None, pose_address=None, exercise_type='', width=constants.POSE_MODEL_WIDTH, height=constants.POSE_MODEL_HEIGHT):
        super().__init__(frame_address, pose_address, exercise_type, width, height, node_name='LocalDisplaySink')

    def write(self, frame):
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

class Mp4FileSink(VideoSink):
    def __init__(self, frame_address=None, pose_address=None, exercise_type='', width=constants.POSE_MODEL_WIDTH, height=constants.POSE_MODEL_HEIGHT, name='video', output_dir=constants.DATA_DIR + '/outputs', fps=30):
        super().__init__(frame_address, pose_address, exercise_type, width, height, node_name='Mp4FileSink')
        self.filename = output_dir + '/' + name + '.mp4'
        self.writer = None
        self.fps = fps

    def prepare(self):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (self.width, self.height))

    def write(self, frame):
        self.writer.write(frame)

    def stop(self):
        print(self.name + ': closing writer to ', self.filename)
        self.writer.release()
        super().stop()
