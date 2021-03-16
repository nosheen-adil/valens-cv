from valens import constants
from valens import sequence
from valens.node import Node
from valens.stream import OutputStream

import h5py

class PoseSource(Node):
    def __init__(self, pose_address, name, input_dir=constants.DATA_DIR + '/sequences'):
        super().__init__("PoseSource")
        self.output_streams["pose"] = OutputStream(pose_address)
        
        self.filename = input_dir + '/' + name + '.h5'
        self.seq = None
        self.t = 0

    def prepare(self):
        with h5py.File(self.filename, 'r') as data:
            self.seq = data['pose'][:]
        # sequence.clean_nans(self.seq)
        # sequence.filter_noise(self.seq)

    def process(self):
        if self.t >= self.seq.shape[-1]:
            self.stop()
            return

        pose = self.seq[:, :, self.t].copy('C')
        self.output_streams["pose"].send(pose)
        self.t += 1
