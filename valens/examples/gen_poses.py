import valens
from valens import constants
from valens.nodes import *
from valens.structures.stream import gen_addr_ipc

import argparse
import time
import torch.multiprocessing
from torch.multiprocessing import set_start_method

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a two stage pipeline to stream video frames')
    parser.add_argument('--input_url', default=0, help='URL for OpenCV VideoCapture')
    parser.add_argument('--filename', default='', help='Filename of .h5 to store sequence. If not provided, then displayed to screen')
    parser.add_argument('--frames', default=100, help='Number of frames of video if storing sequence')
    parser.add_argument('--fps', default=10, help='Maxmimum fps of pipeline')
    args = parser.parse_args()

    torch.multiprocessing.freeze_support()
    set_start_method('spawn')

    frame_address = gen_addr_ipc("frame")
    pose_address = gen_addr_ipc("pose")

    if len(args.filename):
        processes = [VideoSource(
                        frame_address=frame_address,
                        device_url=args.input_url,
                        num_outputs=1),
                    PoseFilter(
                        frame_address=frame_address,
                        pose_address=pose_address),
                    PoseSink(
                        pose_address=pose_address,
                        total_frames=int(args.frames),
                        filename=args.filename)]
    else:
        processes = [VideoSource(
                        frame_address=frame_address,
                        device_url=args.input_url,
                        num_outputs=2),
                    PoseFilter(
                        frame_address=frame_address,
                        pose_address=pose_address),
                    VideoSink(
                            frame_address=frame_address,
                            pose_address=pose_address)]

    processes[0].set_max_fps(int(args.fps))

    for p in processes: p.start()
    for p in processes: p.join()
