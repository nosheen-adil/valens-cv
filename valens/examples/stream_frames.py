import valens
from valens import constants
from valens.nodes import *
from valens.stream import InputStream, OutputStream, gen_addr_ipc, gen_addr_tcp

import argparse
import torch.multiprocessing
from torch.multiprocessing import set_start_method

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a two stage pipeline to stream video frames')
    parser.add_argument('--name', default='', help='Name of input video')
    parser.add_argument('--recordings_dir', default=constants.DATA_DIR + '/recordings', help='Directory for recordings')
    args = parser.parse_args()

    torch.multiprocessing.freeze_support()
    set_start_method('spawn')
    
    processes = [None] * 2
    frame_address = gen_addr_ipc('frame')
    input_url = args.recordings_dir + '/' + args.name + '.mp4'

    processes[0] = VideoSource(
                        frame_address=frame_address,
                        device_url=input_url)
    processes[1] = VideoSink(
                        frame_address=frame_address)

    processes[0].set_max_fps(10.0)

    for process in processes: process.start()
    for process in processes: process.join()
