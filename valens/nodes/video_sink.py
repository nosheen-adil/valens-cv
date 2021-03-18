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

class VideoSink(Node):
    def __init__(self, frame_address=None, pose_address=None, feedback_address=None, right_topology=[], left_topology=[], width=constants.POSE_MODEL_WIDTH, height=constants.POSE_MODEL_HEIGHT, node_name='VideoSink'):
        super().__init__(node_name)

        self.width = width
        self.height = height
        self.right_topology = right_topology
        self.left_topology = left_topology
        if frame_address is not None:
            self.input_streams['frame'] = InputStream(frame_address)
        if pose_address is not None:
            assert(feedback_address is None)
            self.input_streams['pose'] = InputStream(pose_address)
        elif feedback_address is not None:
            self.input_streams['feedback'] = InputStream(feedback_address)

    def process(self):
        if 'frame' in self.input_streams:
            frame, _ = self.input_streams['frame'].recv()
        
            if frame is None:
                if 'pose' in self.input_streams:
                    pose, _ = self.input_streams['pose'].recv()
                    assert(pose is None)
                    self.stop()
                    return
                if 'feedback' in self.input_streams:
                    feedback, _ = self.input_streams['feedback'].recv()
                    assert(feedback is None)
                    self.stop()
                    return
        else:
            frame = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        if 'pose' in self.input_streams:
            pose, _ = self.input_streams['pose'].recv()
            if pose is None:
                self.stop()
                return
            va.pose.draw_on_image(pose, frame, self.right_topology)
        
        elif 'feedback' in self.input_streams:
            feedback, _ = self.input_streams['feedback'].recv()
            if feedback is None:
                self.stop()
                return
            if all([self.right_topology[i][0] in feedback['pose'] for i in range(len(self.right_topology))]):
                va.feedback.draw_on_image(feedback, frame, self.right_topology)
            else:
                assert all([self.left_topology[i][0] in feedback['pose'] for i in range(len(self.left_topology))])
                va.feedback.draw_on_image(feedback, frame, self.left_topology)
        self.write(frame)

    @abstractmethod
    def write(self, frame):
        pass

class LocalDisplaySink(VideoSink):
    def __init__(self, frame_address=None, pose_address=None, feedback_address=None, right_topology=[], left_topology=[], width=constants.POSE_MODEL_WIDTH, height=constants.POSE_MODEL_HEIGHT):
        super().__init__(frame_address=frame_address, pose_address=pose_address, feedback_address=feedback_address, right_topology=right_topology, left_topology=left_topology, width=width, height=height, node_name='LocalDisplaySink')

    def write(self, frame):
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

class Mp4FileSink(VideoSink):
    def __init__(self, frame_address=None, pose_address=None, feedback_address=None, right_topology=[], left_topology=[], width=constants.POSE_MODEL_WIDTH, height=constants.POSE_MODEL_HEIGHT, name='video', output_dir=constants.DATA_DIR + '/outputs', fps=30):
        super().__init__(frame_address=frame_address, pose_address=pose_address, feedback_address=feedback_address, right_topology=right_topology, left_topology=left_topology, width=width, height=height, node_name='Mp4FileSink')
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
