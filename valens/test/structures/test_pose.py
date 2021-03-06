from valens import constants
import valens.structures as core

from valens.structures import sequence
from valens.structures.pose import Keypoints
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

def test_pose_topology_fullset():
    topology = core.pose.topology()
    assert topology == [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [17, 0], [17, 5], [17, 6], [17, 11], [17, 12]]

def test_pose_topology_subset():
    keypoints = [Keypoints.RHIP, Keypoints.RKNEE, Keypoints.RANKLE, Keypoints.NECK]
    topology = core.pose.topology(keypoints)
    assert topology == [[2, 1], [1, 0], [3, 0]]

def test_pose_trt_to_image():
    with h5py.File(constants.TEST_DATA_DIR + "/structures_pose_trt_to_image.h5", "r") as data:
        counts = data["counts"][:]
        objects = data["objects"][:]
        peaks = data["peaks"][:]
        expected_frame = data["frame"][:]
    
    topology = core.pose.topology()

    p = core.pose.nn_to_numpy(counts, objects, peaks)

    capture = cv2.VideoCapture(constants.TEST_DATA_DIR + "/BS_good_Jan15_1_20.mp4")
    _, actual_frame = capture.read()
    core.pose.draw_on_image(p, actual_frame, topology)

    np.testing.assert_equal(actual_frame, expected_frame)

def test_pose_video_sink():
    with h5py.File(constants.TEST_DATA_DIR + "/structures_pose_trt_to_image.h5", "r") as data:
        counts = data["counts"][:]
        objects = data["objects"][:]
        peaks = data["peaks"][:]
        p = core.pose.nn_to_numpy(counts, objects, peaks)
    
    capture = cv2.VideoCapture(constants.TEST_DATA_DIR + "/BS_good_Jan15_1_20.mp4")
    _, frame = capture.read()

    total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Total frames:", total_frames)
    
    frame_addr = gen_addr_ipc("frame")
    pose_addr = gen_addr_ipc("pose")

    frame_stream = OutputStream(frame_addr)
    pose_stream = OutputStream(pose_addr)
    sink = LocalDisplaySink(frame_addr, pose_addr, exercise_type='BS')
    sink.start()

    [s.start() for s in [frame_stream, pose_stream]]
    frame_stream.send(frame)
    pose_stream.send(p)

    frame_stream.send(None)
    pose_stream.send(None)
    sink.join()

def test_pose_h5_sink():
    with h5py.File(constants.TEST_DATA_DIR + "/structures_pose_trt_to_image.h5", "r") as data:
        counts = data["counts"][:]
        objects = data["objects"][:]
        peaks = data["peaks"][:]
        p = core.pose.nn_to_numpy(counts, objects, peaks)

    expected_filename = constants.TEST_DATA_DIR + "/BS_good_Jan15_1_20.h5"
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

def test_pose_vectors():
    keypoints = [Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK]
    topology = [[0, 1], [1, 2], [2, 3]] # [ankle -> knee, knee -> hip, hip -> neck]
    pose = np.array([
        [0.5, 0.7],
        [0.5, 0.5],
        [0.5, 0.35],
        [0.5, 0.1]
    ])

    vectors = core.pose.vectors(pose, topology)

    assert vectors.shape == (3, 2)
    for i in range(3):
        np.testing.assert_equal(vectors[i, :], [0, -1])

def test_pose_angles():
    keypoints = [Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK]
    topology = [[0, 1], [1, 2], [2, 3]] # [ankle -> knee, knee -> hip, hip -> neck]
    pose = np.array([
        [0.7, 0.7],
        [0.7, 0.5],
        [0.5, 0.5],
        [0.5, 0.1]
    ])

    vectors = core.pose.vectors(pose, topology)
    vectors = np.vstack(([0, 1], vectors))
    angles = core.pose.angles(vectors)

    assert angles.shape == (3,)
    assert angles[0] == np.pi
    assert angles[1] == np.pi / 2
    assert angles[2] == np.pi / 2

def test_pose_transform_zero():
    keypoints = [Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK]
    topology = [[0, 1], [1, 2], [2, 3]] # [ankle -> knee, knee -> hip, hip -> neck]
    pose = np.array([
        [0.5, 0.7],
        [0.5, 0.5],
        [0.5, 0.35],
        [0.5, 0.1]
    ])

    joint_angles = np.zeros((4,))
    transformed_pose = core.pose.transform(pose, joint_angles, topology)
    np.testing.assert_allclose(transformed_pose, pose)

def test_pose_transform_nonzero():
    keypoints = [Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK]
    topology = [[0, 1], [1, 2], [2, 3]] # [ankle -> knee, knee -> hip, hip -> neck]
    pose = np.array([
        [0.5, 0.7],
        [0.5, 0.5],
        [0.5, 0.35],
        [0.5, 0.1]
    ])

    expected_pose = pose.copy()
    expected_pose[2, 0] -= abs(pose[2, 1] - pose[1, 1])
    expected_pose[2, 1] = pose[1, 1]
    expected_pose[3, 1] += abs(pose[2, 1] - pose[1, 1])
    expected_pose[3, 0] = expected_pose[2, 0]

    vectors = core.pose.vectors(expected_pose, topology)
    vectors = np.vstack(([0, 1], vectors))
    angles = core.pose.angles(vectors)
    
    joint_angles = np.array([angles[0] - np.pi, -angles[1], angles[2], 0])
    transformed_pose = core.pose.transform(pose, joint_angles, topology)

    np.testing.assert_equal(transformed_pose, expected_pose)

def test_pose_transform_link_length():
    keypoints = [Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK]
    topology = [[0, 1], [1, 2], [2, 3]] # [ankle -> knee, knee -> hip, hip -> neck]
    pose = np.array([
        [0.5, 0.7],
        [0.5, 0.5],
        [0.5, 0.35],
        [0.5, 0.1]
    ])

    joint_angles = np.array([0.1, 0.2, 0.3, 0.4])
    transformed_pose = core.pose.transform(pose, joint_angles, topology)

    for i in range(1, joint_angles.shape[0]):
        joint1 = topology[i-1][0]
        joint2 = topology[i-1][1]

        dist1 = np.linalg.norm(pose[joint2, :] - pose[joint1, :])
        dist2 = np.linalg.norm(transformed_pose[joint2, :] - transformed_pose[joint1, :])
        
        np.testing.assert_allclose(dist1, dist2)
