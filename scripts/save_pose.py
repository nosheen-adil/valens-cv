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
    # sequences_dir = va.constants.DATA_DIR + '/sequences'
    # recordings_dir = va.constants.DATA_DIR + '/recordings'

    proxy = Process(target=va.bus.proxy)
    proxy.start()
    bus = va.bus.MessageBus()
    bus.subscribe('finished')
    nodes = [
        VideoSource(is_live=True),
        PoseFilter(),
        PoseSink(sequences_dir=sequences_dir)
    ]

    for node in nodes:
        node.start()
    time.sleep(60)

    # _, _, names = next(os.walk(recordings_dir))
    # names = [os.path.splitext(os.path.basename(name))[0] for name in names]
    names = ['BS_bad_4']
    for i, name in enumerate(names):
        print('Case:', name)
        request = {
            'signal' : 'start',
            'user_id' : 'test',
            'set_id' : 'set_' + str(i),
            'exercise' : name[0:2],
            'num_reps' : -1,
            'capture' : recordings_dir + '/' + name + '.mp4',
            'name' : name,
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
