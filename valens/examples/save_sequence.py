import valens
from valens import constants
from valens.nodes import *
from valens.stream import gen_addr_ipc

import argparse
import cv2
import time
import torch.multiprocessing
from torch.multiprocessing import set_start_method

def total_frames(url):
    c = cv2.VideoCapture(url)
    return int(c.get(cv2.CAP_PROP_FRAME_COUNT))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a three stage pipeline to save pose sequences to an .h5 file')
    parser.add_argument('--name', default='', help='Name of input video')
    parser.add_argument('--recordings_dir', default=constants.DATA_DIR + '/recordings', help='Directory for recordings')
    parser.add_argument('--sequences_dir', default=constants.DATA_DIR + '/sequences', help='Directory for sequences')
    args = parser.parse_args()

    torch.multiprocessing.freeze_support()
    set_start_method('spawn')

    frame_address = gen_addr_ipc("frame")
    pose_address = gen_addr_ipc("pose")

    input_url = args.recordings_dir + '/' + args.name + '.mp4'
    output_filename = args.sequences_dir + '/' + args.name + '.h5'
    processes = [VideoSource(
                    frame_address=frame_address,
                    device_url=input_url,
                    num_outputs=1),
                PoseFilter(
                    frame_address=frame_address,
                    pose_address=pose_address),
                PoseSink(
                    total_frames=total_frames(input_url),
                    filename=output_filename,
                    pose_address=pose_address)]


    for p in processes: p.start()
    for p in processes: p.join()
