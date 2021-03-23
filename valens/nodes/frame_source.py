from valens import constants
from valens.node import Node
#from valens.stream import OutputStream, gen_set_id, gen_sync_metadata, gen_addr_ipc
from valens.exercise import ExerciseType
from valens.stream import gen_addr_tcp

import cv2
import zmq

class FrameSource(Node):
    def __init__(self, max_fps=10, frame_port=7000):
        super().__init__("FrameSource")
        self.set_max_fps(max_fps)

        self.user_id = None
        self.exercise = None
        self.set_id = None
        self.context = zmq.Context()
        self.frame_socket = self.context.socket(zmq.REQ)
        self.frame_socket.connect(gen_addr_tcp(frame_port, bind=False))

    def reset(self):
        self.user_id = None
        self.exercise = None
        self.set_id = None

    def configure(self, request):
        self.user_id = request['user_id']
        self.exercise = str(ExerciseType(request['exercise']))
        self.set_id = request['set_id']

    def process(self):
        request = "req"
        self.frame_socket.send_json(request)
        print('sent request')
        response = self.frame_socket.recv_json()
        print('got response', response)

