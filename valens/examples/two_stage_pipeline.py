import valens
from valens.structures.stream import InputStream, OutputStream, gen_addr_ipc, gen_addr_tcp
from valens.nodes import *
from valens import constants

import torch.multiprocessing
from torch.multiprocessing import set_start_method

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    set_start_method('spawn')
    
    processes = [None] * 2
    frame_address = gen_addr_ipc("test")
    video_path = constants.DATA_DIR + "/yoga_5min.mp4"

    processes[0] = VideoSource(
                        frame_address=frame_address,
                        device_url=video_path)
    processes[1] = VideoSink(
                        frame_address=frame_address)

    processes[0].set_max_fps(15)

    for process in processes: process.start()
    for process in processes: process.join()

