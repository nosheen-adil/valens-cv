import valens as va
from valens.nodes import *
import valens.bus
import valens.constants

import cv2
import time
import torch.multiprocessing
from torch.multiprocessing import set_start_method
from torch.multiprocessing import Process
import os

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    set_start_method('spawn')

    sequences_dir = va.constants.DATA_DIR + '/sequences/shashank'
    recordings_dir = va.constants.DATA_DIR + '/recordings/shashank'
    outputs_dir = va.constants.DATA_DIR + '/outputs/shashank'

    proxy = Process(target=va.bus.proxy)
    proxy.start()
    bus = va.bus.MessageBus()
    bus.subscribe('finished')
    nodes = [
        VideoSource(max_fps=15, num_outputs=2),
        PoseFilter(),
        FeedbackFilter(output_frame=True, output_feedback=False, is_live=True),
        Mp4FileSink(output_dir=outputs_dir)
    ]

    for node in nodes:
        node.start()
    time.sleep(10)

    # _, _, names = next(os.walk(recordings_dir))
    # names = [os.path.splitext(os.path.basename(name))[0] for name in names]
    names = ['BC_bad_3', 'BC_bad_4', 'BS_bad_2', 'BS_bad_3', 'BC_bad_1', 'BC_bad_2', 'BC_good_1', 'BC_good_2', 'BS_bad_1', 'BS_bad_4', 'BS_good_1', 'BS_good_2', 'PU_good_1', 'PU_good_2', 'PU_bad_1', 'PU_bad_2', 'PU_bad_3', 'PU_bad_4']
    for i, name in enumerate(names):
        print('Case:', name)
        request = {
            'signal' : 'start',
            'user_id' : 'demo_3',
            'set_id' : 'set_' + str(i),
            'exercise' : name[0:2],
            'num_reps' : 5,
            'capture' : recordings_dir + '/' + name + '.mp4',
            'name' : name,
            'publish' : True
        }
        bus.send('signal', request)
        finished = bus.recv('finished', timeout=None)
        assert finished is True

    request = {
        'signal' : 'stop'
    }
    bus.send('signal', request)

    for node in nodes:
        node.join()

    proxy.terminate()
