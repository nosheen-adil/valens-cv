import valens
from valens import constants
from valens.nodes import *
from valens.stream import gen_addr_ipc
import valens as va
import valens.pose
from valens.pose import Keypoints

import argparse
import time
import torch.multiprocessing
from torch.multiprocessing import set_start_method

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a three stage pipeline to stream pose sequences to the local display')
    parser.add_argument('--input', default='', help='Name of input video, if not saved to a sequence, provide full path')
    parser.add_argument('--saved', action='store_true', help='Indicates if the sequence is already saved')
    parser.add_argument('--mp4', action='store_true', help='Save video to an mp4 file')
    parser.add_argument('--fps', default=30, help='Fps of ouput video if saved to mp4')
    parser.add_argument('--recordings_dir', default=constants.DATA_DIR + '/recordings', help='Directory for recordings')
    parser.add_argument('--sequences_dir', default=constants.DATA_DIR + '/sequences', help='Directory for saved sequences')
    parser.add_argument('--outputs_dir', default=constants.DATA_DIR + '/outputs', help='Directory to store output videos')
    args = parser.parse_args()

    torch.multiprocessing.freeze_support()
    set_start_method('spawn')

    frame_address = gen_addr_ipc("frame")
    pose_address = gen_addr_ipc("pose")
    exercise_type = args.input[0:2]

    keypoints = [Keypoints.NECK, Keypoints.RSHOULDER, Keypoints.RHIP, Keypoints.RKNEE, Keypoints.RANKLE, Keypoints.RELBOW, Keypoints.RWRIST]
    topology = va.pose.topology(keypoints)
    if args.mp4:
        video_sink = Mp4FileSink(
            frame_address=frame_address,
            pose_address=pose_address,
            left_topology=topology,
            right_topology=topology,
            name=args.input,
            output_dir=args.outputs_dir,
            fps=int(args.fps))
    else:
        video_sink = LocalDisplaySink(
            frame_address=frame_address,
            pose_address=pose_address,
            left_topology=topology,
            right_topology=topology,)
        video_sink.set_max_fps(int(args.fps))

    input_url = args.recordings_dir + '/' + args.input + '.mp4'
    if args.saved:
        input_file = args.sequences_dir + '/' + args.input + '.h5'
        processes = [VideoSource(
                        frame_address=frame_address,
                        device_url=int(args.input) if args.input.isdigit() else input_url,
                        num_outputs=1),
                    PoseSource(
                        pose_address=pose_address,
                        name=args.input,
                        input_dir=args.sequences_dir),
                    video_sink]
    else:
        processes = [VideoSource(
                        frame_address=frame_address,
                        device_url=int(args.input) if args.input.isdigit() else input_url,
                        num_outputs=2),
                    PoseFilter(
                        frame_address=frame_address,
                        pose_address=pose_address),
                    video_sink]

    for p in processes: p.start()
    for p in processes: p.join()
