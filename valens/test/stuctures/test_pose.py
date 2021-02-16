import pytest
import h5py
import cv2
import trt_pose.coco
from trt_pose.draw_objects import DrawObjects
import json
import numpy as np

from valens import constants
from valens.structures import pose

def test_trt_to_image():
    with h5py.File(constants.TEST_DATA_DIR + "/structures_pose_trt_to_image.h5", "r") as data:
        counts = data["counts"][:]
        objects = data["objects"][:]
        peaks = data["peaks"][:]
        frame = data["frame"][:]
    
    topology = pose.get_topology()

    p = pose.nn_to_numpy(counts, objects, peaks)
    p = pose.scale(p, constants.POSE_MODEL_WIDTH, constants.POSE_MODEL_HEIGHT, round_coords=True)
    actual_frame = frame.copy()
    pose.draw_on_image(p, actual_frame, topology)

    np.testing.assert_equal(actual_frame, frame)
