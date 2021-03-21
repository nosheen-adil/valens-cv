from valens import constants
from valens.node import Node
from valens.stream import OutputStream, gen_set_id, gen_sync_metadata, gen_addr_ipc
from valens.exercise import ExerciseType

import h5py

class PoseSource(Node):
    def __init__(self, pose_address=gen_addr_ipc("pose"), max_fps=10, input_dir=constants.DATA_DIR + '/sequences'):
        super().__init__("PoseSource")

        self.output_streams["pose"] = OutputStream(pose_address)
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
        self.set_id = gen_set_id(16)
        self.diff_frames = round(int(request['original_fps']) / self.max_fps) - 1
        
        filename = self.input_dir + '/' + request['name'] + '.h5'
        with h5py.File(filename, 'r') as data:
            self.seq = data['pose'][:]

    def process(self):
        if self.t >= self.seq.shape[-1]:
            # self.stop()
            self.bus.send("finished")
            return

        pose = self.seq[:, :, self.t].copy('C')
        sync = gen_sync_metadata(self.user_id, self.exercise, self.set_id)
        self.output_streams["pose"].send(pose, sync)
        self.t += 1

        for _ in range(self.diff_frames):
            self.t += 1
