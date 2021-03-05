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
    parser.add_argument('--input', default='', help='Name of input video, if not saved to a sequence, provide full path')
    parser.add_argument('--saved', action='store_true', help='Indicates if the sequence is already saved')
    parser.add_argument('--mp4', action='store_true', help='Save video to an mp4 file')
    parser.add_argument('--fps', default=30, help='Fps of ouput video if saved to mp4')
    parser.add_argument('--recordings_dir', default=constants.DATA_DIR + '/recordings', help='Directory for recordings')
    parser.add_argument('--sequences_dir', default=constants.DATA_DIR + '/sequences', help='Directory for saved sequences')
    parser.add_argument('--outputs_dir', default=constants.DATA_DIR + '/outputs/aligned', help='Directory to store output videos')
    args = parser.parse_args()

    torch.multiprocessing.freeze_support()
    set_start_method('spawn')

    pose_address = gen_addr_ipc("pose")
    feedback_address = gen_addr_ipc("feedback")
    exercise_type = args.input[0:2]
    
    if args.mp4:
        video_sink = Mp4FileSink(
            pose_address=feedback_address,
            exercise_type=exercise_type,
            name=args.input,
            output_dir=args.outputs_dir,
            fps=int(args.fps))
    else:
        video_sink = LocalDisplaySink(
            pose_address=feedback_address,
            exercise_type=exercise_type)
        video_sink.set_max_fps(int(args.fps))

    processes = [PoseSource(
                    pose_address=pose_address,
                    name=args.input,
                    input_dir=args.sequences_dir),
                FeedbackFilter(
                    pose_address=pose_address,
                    feedback_address=feedback_address),
                video_sink]

    for p in processes: p.start()
    for p in processes: p.join()