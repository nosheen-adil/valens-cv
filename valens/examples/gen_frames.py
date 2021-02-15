import valens
from valens.structures.stream import InputStream, OutputStream, gen_addr_ipc, gen_addr_tcp
from valens.nodes import *
from valens import constants

import argparse
import torch.multiprocessing
from torch.multiprocessing import set_start_method

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a two stage pipeline to stream video frames')
    parser.add_argument('--input_url', type=str, default=0, help = 'URL for OpenCV VideoCapture')
    args = parser.parse_args()

    torch.multiprocessing.freeze_support()
    set_start_method('spawn')
    
    processes = [None] * 2
    frame_address = gen_addr_ipc('frame')

    processes[0] = VideoSource(
                        frame_address=frame_address,
                        device_url=args.input_url)
    processes[1] = VideoSink(
                        frame_address=frame_address)

    processes[0].set_max_fps(10.0)

    for process in processes: process.start()
    for process in processes: process.join()

