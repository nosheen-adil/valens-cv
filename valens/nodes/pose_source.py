from valens import constants
from valens.node import Node
from valens.stream import OutputStream, gen_set_id, gen_sync_metadata, gen_addr_ipc
from valens.exercise import ExerciseType

import h5py
import cv2

def get_capture_fps(name):
    c = cv2.VideoCapture(name)
    return int(c.get(cv2.CAP_PROP_FPS))

class PoseSource(Node):
    def __init__(self, max_fps=10, input_dir=constants.DATA_DIR + '/sequences'):
        super().__init__("PoseSource")

        self.output_streams["pose"] = OutputStream(gen_addr_ipc('pose'), identities=[b'0'])
        self.input_dir = input_dir
        self.set_max_fps(max_fps)

        self.user_id = None
        self.exercise = None
        self.set_id = None
        self.diff_frames = 0
        self.t = 0
        self.seq = None

    def reset(self):
        self.user_id = None
        self.exercise = None
        self.set_id = None
        self.diff_frames = 0
        self.t = 0
        self.seq = None

    def configure(self, request):
        self.user_id = request['user_id']
        self.exercise = str(ExerciseType(request['exercise']))
        self.set_id = request['set_id']
        self.diff_frames = round(int(get_capture_fps(request['capture'])) / self.max_fps) - 1
        
        filename = self.input_dir + '/' + request['name'] + '.h5'
        with h5py.File(filename, 'r') as data:
            self.seq = data['pose'][:]

    def process(self):
        finished = self.bus.recv('finished')
        if finished is True:
            print('PoseSource: caught finished, forwarding sentinel')
            self.output_streams['pose'].send() # forward sentinel
            return True

        if self.t >= self.seq.shape[-1]:
            print(self.name + ': finished data, forwarding sentinel')
            
            self.bus.send("finished")
            self.output_streams['pose'].send() # forward sentinel
            return True

        # print(self.name + ': sending')
        pose = self.seq[:, :, self.t].copy('C')
        sync = gen_sync_metadata(self.user_id, self.exercise, self.set_id)
        sync['id'] = self.iterations
        sync['fps'] = 0
        # self.average_fps()
        self.output_streams['pose'].send(pose, sync)
        self.t += 1
        # print(self.name + ': sent')

        for _ in range(self.diff_frames):
            self.t += 1
