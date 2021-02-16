import pytest
import h5py
import cv2
import trt_pose.coco
from trt_pose.draw_objects import DrawObjects
import json
import numpy as np

from valens import constants
from valens.structures import pose
from valens.structures.stream import OutputStream, gen_addr_ipc
from valens.nodes import *

def test_trt_to_image():
    with h5py.File(constants.TEST_DATA_DIR + "/structures_pose_trt_to_image.h5", "r") as data:
        counts = data["counts"][:]
        objects = data["objects"][:]
        peaks = data["peaks"][:]
        expected_frame = data["frame"][:]
    
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
    sink.join()
