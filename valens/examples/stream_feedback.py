import valens
from valens import constants
from valens.nodes import *
from valens.stream import gen_addr_ipc

import valens as va
import valens.exercise
import valens.feedback
from valens.pose import Keypoints

import argparse
import cv2
import time
import torch.multiprocessing
from torch.multiprocessing import set_start_method

def get_capture_fps(name):
    c = cv2.VideoCapture(name)
    return int(c.get(cv2.CAP_PROP_FPS))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a three stage pipeline to stream pose sequences to the local display')
    parser.add_argument('--input', default='', help='Name of input video, if not saved to a sequence, provide full path')
    parser.add_argument('--saved', action='store_true', help='Indicates if the sequence is already saved')
    parser.add_argument('--mp4', action='store_true', help='Save video to an mp4 file')
    parser.add_argument('--overlay', action='store_true', help='Save video to an mp4 file')
    parser.add_argument('--fps', default=30, help='Fps of ouput video if saved to mp4')
    parser.add_argument('--recordings_dir', default=constants.DATA_DIR + '/recordings', help='Directory for recordings')
    parser.add_argument('--sequences_dir', default=constants.DATA_DIR + '/sequences', help='Directory for saved sequences')
    parser.add_argument('--feedback_dir', default=constants.DATA_DIR + '/feedback', help='Directory to store feedback videos')
    args = parser.parse_args()

    torch.multiprocessing.freeze_support()
    set_start_method('spawn')

    pose_address = gen_addr_ipc("pose")
    feedback_address = gen_addr_ipc("feedback")
    exercise_type = args.input[0:2]
    exercise = va.exercise.load(exercise_type)

    if args.mp4:
        if args.overlay:
            video_sink = Mp4FileSink(
                frame_address=gen_addr_ipc("frame"),
                feedback_address=feedback_address,
                right_topology=exercise.topology(va.exercise.Side.RIGHT),left_topology=exercise.topology(va.exercise.Side.LEFT),
                name=args.input,
                output_dir=args.feedback_dir,
                fps=int(args.fps))
        else:
            video_sink = Mp4FileSink(
                feedback_address=feedback_address,
                right_topology=exercise.topology(va.exercise.Side.RIGHT),left_topology=exercise.topology(va.exercise.Side.LEFT),
                name=args.input,
                output_dir=args.feedback_dir,
                fps=int(args.fps))
    else:
        if args.overlay:
            video_sink = LocalDisplaySink(
                frame_address=gen_addr_ipc("frame"),
                feedback_address=feedback_address,
                right_topology=exercise.topology(va.exercise.Side.RIGHT),left_topology=exercise.topology(va.exercise.Side.LEFT))
        else:
            video_sink = LocalDisplaySink(
                feedback_address=feedback_address,
                right_topology=exercise.topology(va.exercise.Side.RIGHT),left_topology=exercise.topology(va.exercise.Side.LEFT))
        video_sink.set_max_fps(int(args.fps))

    if args.overlay:
        processes = [
                    VideoSource(
                        user_id="nosheen123",
                        exercise_type=exercise_type,
                        frame_address=gen_addr_ipc('frame'),
                        device_url=args.recordings_dir + '/' + args.input + '.mp4',
                        max_fps=int(args.fps)),
                    PoseSource(
                        user_id="nosheen123",
                        name=args.input,
                        pose_address=pose_address,
                        input_dir=args.sequences_dir,
                        original_fps=get_capture_fps(args.recordings_dir + '/' + args.input + '.mp4'),
                        max_fps=int(args.fps)),
                    FeedbackFilter(
                        pose_address=pose_address,
                        feedback_address=feedback_address,
                        exercise=exercise),
                    video_sink]
    else:
        processes = [PoseSource(
                        user_id="nosheen123",
                        name=args.input,
                        pose_address=pose_address,
                        input_dir=args.sequences_dir),
                    FeedbackFilter(
                        pose_address=pose_address,
                        feedback_address=feedback_address,
                        exercise=exercise),
                    video_sink]

    for p in processes: p.start()
    for p in processes: p.join()