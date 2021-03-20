from valens import constants
from valens.node import Node
from valens.stream import OutputStream, gen_set_id, gen_sync_metadata, gen_addr_ipc
from valens.exercise import ExerciseType

import h5py

class PoseSource(Node):
    def __init__(self, name, user_id='user', set_id=gen_set_id(16), pose_address=gen_addr_ipc("pose"), original_fps=30, max_fps=10, input_dir=constants.DATA_DIR + '/sequences'):
        super().__init__("PoseSource")

        self.user_id = user_id
        self.exercise = str(ExerciseType(name[0:2]))
        self.set_id = set_id
        self.output_streams["pose"] = OutputStream(pose_address)
        
        self.filename = input_dir + '/' + name + '.h5'
        self.seq = None
        self.t = 0

        self.set_max_fps(max_fps)
        self.diff_frames = round(original_fps / max_fps) - 1
        print(self.name, 'diff frames:', self.diff_frames)

    def prepare(self):
        with h5py.File(self.filename, 'r') as data:
            self.seq = data['pose'][:]

    def process(self):
        if self.t >= self.seq.shape[-1]:
            self.stop()
            return

        pose = self.seq[:, :, self.t].copy('C')
        sync = gen_sync_metadata(self.user_id, self.exercise, self.set_id)
        self.output_streams["pose"].send(pose, sync)
        self.t += 1

        for _ in range(self.diff_frames):
            self.t += 1
