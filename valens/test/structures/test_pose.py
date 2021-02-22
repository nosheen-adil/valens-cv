from valens import constants
from valens.structures import pose, sequence
from valens.structures.stream import OutputStream, gen_addr_ipc
from valens.nodes import *

import cv2
import h5py
import json
import numpy as np
import os
import pytest
import trt_pose.coco
from trt_pose.draw_objects import DrawObjects

def test_trt_to_image():
    with h5py.File(constants.TEST_DATA_DIR + "/structures_pose_trt_to_image.h5", "r") as data:
        counts = data["counts"][:]
        objects = data["objects"][:]
        peaks = data["peaks"][:]
        expected_frame = data["frame"][:]
        print(expected_frame.shape, expected_frame.dtype)
    
    topology = pose.get_topology()

    p = pose.nn_to_numpy(counts, objects, peaks)

    capture = cv2.VideoCapture(constants.DATA_DIR + "/recordings/good_BS_Jan15_1_20.mp4")
    _, actual_frame = capture.read()
    pose.draw_on_image(p, actual_frame, topology)

    np.testing.assert_equal(actual_frame, expected_frame)

def test_video_sink():
    with h5py.File(constants.TEST_DATA_DIR + "/structures_pose_trt_to_image.h5", "r") as data:
        counts = data["counts"][:]
        objects = data["objects"][:]
        peaks = data["peaks"][:]
        p = pose.nn_to_numpy(counts, objects, peaks)
    
    capture = cv2.VideoCapture(constants.DATA_DIR + "/recordings/good_BS_Jan15_1_20.mp4")
    _, frame = capture.read()

    total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Total frames:", total_frames)
    
    frame_addr = gen_addr_ipc("frame")
    pose_addr = gen_addr_ipc("pose")

    frame_stream = OutputStream(frame_addr)
    pose_stream = OutputStream(pose_addr)
    sink = VideoSink(frame_addr, pose_addr)
    sink.start()

    [s.start() for s in [frame_stream, pose_stream]]
    frame_stream.send(frame)
    pose_stream.send(p)

    frame_stream.send(None)
    pose_stream.send(None)
    sink.join()

def test_pose_sink():
    with h5py.File(constants.TEST_DATA_DIR + "/structures_pose_trt_to_image.h5", "r") as data:
        counts = data["counts"][:]
        objects = data["objects"][:]
        peaks = data["peaks"][:]
        p = pose.nn_to_numpy(counts, objects, peaks)

    expected_filename = constants.DATA_DIR + "/sequences/good_BS_Jan15_1_20.h5"
    actual_filename = constants.TEST_DATA_DIR + "/tmp_pose_sink.h5"

    pose_addr = gen_addr_ipc("pose")
    pose_stream = OutputStream(pose_addr)
    sink = PoseSink(pose_addr, total_frames=1, filename=actual_filename)
    sink.start()

    pose_stream.start()
    pose_stream.send(p)
    pose_stream.send(None)
    sink.join()

    with h5py.File(expected_filename) as f:
        expected = f["pose"][:]

    with h5py.File(actual_filename) as f:
        actual = f["pose"][:]

    np.testing.assert_equal(actual[:, :, 0], expected[:, :, 0])
    os.remove(actual_filename)
