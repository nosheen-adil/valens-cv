import valens as va
from valens import constants
from valens.node import Node
from valens.stream import InputStream, OutputStream, gen_addr_ipc
import valens.exercise

import numpy as np
from abc import ABC, abstractmethod
import time

class FeedbackFilter(Node):
    def __init__(self, pose_address=gen_addr_ipc('pose'), feedback_address=gen_addr_ipc('feedback')):
        super().__init__('FeedbackFilter', topics=['pose'])

        # self.input_streams['pose'] = InputStream(pose_address)
        # self.output_streams['feedback'] = OutputStream(feedback_address)
        self.exercise = None
        self.num_reps = None
        self.set_id = None

    def reset(self):
        self.exercise = None
        self.num_reps = None
        self.set_id = None

    def configure(self, request):
        self.exercise = va.exercise.load(request['exercise'])
        self.num_reps = request['num_reps']
        self.set_id = request['set_id']

    def process(self):
        # data = self.input_streams['pose'].recv(timeout=)
        data = self.bus.recv("pose", timeout=self.timeout)
        if data is False:
            return
        sync, pose = data

        if sync['set_id'] != self.set_id:
            print(self.name + ': mismatched frame and set, discarding...')
            return

        self.exercise.fit(pose)
        feedback = self.exercise.predict()
        if feedback is not None:
            # self.output_streams['feedback'].send(feedback, sync)
            self.bus.send('feedback', feedback, sync)
        # print(self.name + ': sending feedback', sync['id'])

        if self.exercise.reps == self.num_reps:
            print(self.name + ': max number of reps detected, stopping...')
            self.bus.send('feedback_finished')
            # feedback = self.exercise.predict()
            # self.bus.send('feedback', feedback, sync)
            return True

        
