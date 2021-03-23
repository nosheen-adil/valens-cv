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
    def __init__(self, topics=['frame'], width=constants.POSE_MODEL_WIDTH, height=constants.POSE_MODEL_HEIGHT, node_name='VideoSink'):
        super().__init__(node_name, topics=topics + ['feedback_finished'])

        assert set(topics).issubset(set(['frame', 'pose', 'feedback']))

        self.width = width
        self.height = height
        self.right_topology = None
        self.left_topology = None
        self.set_id = None
        
        # if frame_address is not None:
        #     self.input_streams['frame'] = InputStream(frame_address)
        # if pose_address is not None:
        #     assert(feedback_address is None)
        #     self.input_streams['pose'] = InputStream(pose_address)
        # elif feedback_address is not None:
        #     self.input_streams['feedback'] = InputStream(feedback_address)

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
        data = self.bus.recv('feedback', timeout=self.timeout if self.iterations > 0 else None)
        if data is False:
            feedback_finished = self.bus.recv('feedback_finished', timeout=self.timeout)
            if feedback_finished is False:
                return
            else:
                print(self.name + ': detected feedback finished')
                assert feedback_finished is True
                self.bus.send('finished')
                return
        feedback_sync, feedback = data

        if 'frame' in self.topics:
            data = self.bus.recv('frame', timeout=self.timeout)
            if data is False:
                return
            frame_sync, frame = data
            # print('set_id', self.set_id, feedback_sync['set_id'], frame_sync['set_id'], feedback_sync['id'], frame_sync['id'])

            if feedback_sync['id'] < frame_sync['id']:
                if feedback_sync['set_id'] == self.set_id:
                    print('SKIPping')
                    if frame_sync['set_id'] != self.set_id:
                        while frame_sync['set_id'] != self.set_id: # skip frames from old set
                            data = self.bus.recv('frame', timeout=None)
                            frame_sync, frame = data
                        # print('after: set_id', self.set_id, feedback_sync['set_id'], frame_sync['set_id'], feedback_sync['id'], frame_sync['id'])
                    else:
                        return
                else:
                    return # don't want feedback not corresponding to this set
            # if feedback_sync['id'] > frame_sync['id']:
            #     while feedback_sync['id'] > frame_sync['id']:
            #         data = self.bus.recv('frame', timeout=None)
            #         frame_sync, frame = data
        else:
            frame = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        # print('set_id', self.set_id, feedback_sync['set_id'], frame_sync['set_id'], feedback_sync['id'], frame_sync['id'])

        # if self.set_id ==

        if all([self.right_topology[i][0] in feedback['pose'] for i in range(len(self.right_topology))]):
            va.feedback.draw_on_image(feedback, frame, self.right_topology)
        elif all([self.left_topology[i][0] in feedback['pose'] for i in range(len(self.left_topology))]):
            va.feedback.draw_on_image(feedback, frame, self.left_topology)
        else:
            print(self.name + ': topology does not match')
            return
        self.write(frame)

    @abstractmethod
    def write(self, frame):
        pass

class LocalDisplaySink(VideoSink):
    def __init__(self, topics=['frame']):
        super().__init__(topics=topics, node_name='LocalDisplaySink')

    def write(self, frame):
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

class Mp4FileSink(VideoSink):
    def __init__(self, topics=['frame'], output_dir=constants.DATA_DIR + '/outputs', fps=10):
        super().__init__(topics=topics, node_name='Mp4FileSink')
        self.output_dir = output_dir
        self.fps = fps
        self.filename = None
        self.writer = None

    def reset(self):
        print(self.name + ': closing writer to ', self.filename)
        self.writer.release()

        self.filename = None
        self.writer = None
        super().reset()

    def configure(self, request):
        super().configure(request)

        self.filename = self.output_dir + '/' + request['name'] + '.mp4'
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (self.width, self.height))

    def write(self, frame):
        self.writer.write(frame)
