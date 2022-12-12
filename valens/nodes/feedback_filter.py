import valens as va
from valens import constants
from valens.node import Node
from valens.stream import InputStream, OutputStream, gen_addr_ipc, gen_addr_tcp
import valens.exercise
import valens.feedback

import numpy as np
from abc import ABC, abstractmethod
import time

class FeedbackFilter(Node):
    def __init__(self, output_frame=True, output_feedback=False, is_live=True):
        super().__init__('FeedbackFilter')
        self.exercise = None
        self.num_reps = None
        self.set_id = None
        self.right_topology = None
        self.left_topology = None
        
        self.input_streams['pose'] = InputStream(gen_addr_ipc('pose'), identity=b'0')
        if output_frame:
            self.input_streams['frame'] = InputStream(gen_addr_ipc('frame'), identity=b'1' if is_live else b'0')
            # self.output_streams['feedback_frame'] = OutputStream(gen_addr_tcp(7000, bind=True), identities=[b'0'])
        if output_feedback:
            self.output_streams['feedback'] = OutputStream(gen_addr_ipc('feedback'), identities=[b'0'])

    def reset(self):
        self.exercise = None
        self.num_reps = None
        self.set_id = None
        self.right_topology = None
        self.left_topology = None

    def configure(self, request):
        self.exercise = va.exercise.load(request['exercise'])
        self.num_reps = int(request['num_reps'])
        self.set_id = request['set_id']
        self.right_topology = self.exercise.topology(side=va.exercise.Side.RIGHT)
        self.left_topology = self.exercise.topology(side=va.exercise.Side.LEFT)

    def process(self):
        # print('getting pose', self.iterations)
        if 'frame' in self.input_streams:
            frame, frame_sync = self.input_streams['frame'].recv()
            if frame is None:
                print('FeedbackFilter: detected frame sentinel')
                pose, pose_sync = self.input_streams['pose'].recv()
                print('FeedbackFilter: got pose sentinel')
                for s in self.output_streams.values():
                    s.send()
                return True

        pose, pose_sync = self.input_streams['pose'].recv()
        if pose is None:
            print('FeedbackFilter: detected pose sentinel')
            for s in self.output_streams.values():
                s.send()
            return True

        self.exercise.fit(pose)
        feedback = self.exercise.predict()
        # print('got feedback', feedback)
        if feedback is not None:
            if 'frame' in self.input_streams:
                if all([self.right_topology[i][0] in feedback['pose'] for i in range(len(self.right_topology))]):
                    va.feedback.draw_on_image(feedback, frame, self.right_topology, fps=pose_sync['fps'])
                elif all([self.left_topology[i][0] in feedback['pose'] for i in range(len(self.left_topology))]):
                    va.feedback.draw_on_image(feedback, frame, self.left_topology, fps=pose_sync['fps'])
                else:
                    print(self.name + ': topology does not match')
                    return

                # print(self.name + ': sending')
                self.bus.send('feedback_frame', frame)
                # print(self.name + ': sent')
            
            if 'feedback' in self.output_streams:
                self.output_streams['feedback'].send(feedback, frame_sync)

        if self.exercise.reps == self.num_reps:
            print(self.name + ': max number of reps detected, stopping...')
            
            self.bus.send('finished')
            
            while True:
                frame, sync = self.input_streams['frame'].recv()
                pose, sync = self.input_streams['pose'].recv()
                if frame is None:
                    assert pose is None
                    break

            for s in self.output_streams.values():
                s.send()
            
            return True
