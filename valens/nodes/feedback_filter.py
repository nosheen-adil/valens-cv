from valens import constants
from valens.structures.node import Node
from valens.structures.stream import InputStream, OutputStream, gen_addr_ipc
from valens.structures import sequence

import numpy as np
from abc import ABC, abstractmethod


class FeedbackFilter(Node):
    def __init__(self, exercise, pose_address=gen_addr_ipc('pose'), feedback_address=gen_addr_ipc('feedback')):
        super().__init__('FeedbackFilter')

        self.input_streams['pose'] = InputStream(pose_address)
        self.output_streams['feedback'] = OutputStream(feedback_address)
        self.exercise = exercise
        self.exercise.load_classifier()

    def process(self):
        pose = self.input_streams['pose'].recv()
        if pose is None:
            self.stop()
            return

        self.exercise.fit(pose)
        feedback = self.exercise.predict()
        if feedback is not None:
            self.output_streams['feedback'].send(feedback)
