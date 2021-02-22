import valens
from valens import constants
from valens.nodes import *
from valens.structures.stream import gen_addr_ipc

import argparse
import time
import torch.multiprocessing
from torch.multiprocessing import set_start_method

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a three stage pipeline to stream pose sequences to the local display')
    parser.add_argument('--name', default='', help='Name of input video')
    parser.add_argument('--recordings_dir', default=constants.DATA_DIR + '/recordings', help='Directory for recordings')
    args = parser.parse_args()

    torch.multiprocessing.freeze_support()
    set_start_method('spawn')

    frame_address = gen_addr_ipc("frame")
    pose_address = gen_addr_ipc("pose")

    input_url = args.recordings_dir + '/' + args.name + '.mp4'
    processes = [VideoSource(
                    frame_address=frame_address,
                    device_url=input_url,
                    num_outputs=2),
                PoseFilter(
                    frame_address=frame_address,
                    pose_address=pose_address),
                VideoSink(
                        frame_address=frame_address,
                        pose_address=pose_address)]

    for p in processes: p.start()
    for p in processes: p.join()
