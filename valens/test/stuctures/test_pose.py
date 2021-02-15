import pytest
import h5py
import cv2
import trt_pose.coco
import json
import numpy as np

from valens import constants
from valens.structures import pose

def test_trt_to_image():
    with h5py.File(constants.TEST_DATA_DIR + "/structures_trt_to_image.h5", "r") as data:
        counts = data["counts"][:]
        objects = data["objects"][:]
        peaks = data["peaks"][:]
        frame = data["frame"][:]
    
    j = pose.trt_to_json(counts, objects, peaks)
    pose.scale(j, constants.POSE_MODEL_WIDTH, constants.POSE_MODEL_HEIGHT)

    topology = pose.get_topology(constants.DATA_DIR + "/human_pose.json")

    capture = cv2.VideoCapture(constants.DATA_DIR + "/yoga_5min.mp4")
    ret, actual_frame = capture.read()
    actual_frame = cv2.resize(actual_frame, dsize=(constants.POSE_MODEL_WIDTH, constants.POSE_MODEL_HEIGHT))
    pose.draw_on_image(topology, actual_frame, j)

    np.testing.assert_equal(actual_frame, frame)
