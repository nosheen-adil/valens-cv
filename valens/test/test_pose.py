import valens as va
from valens.pose import Keypoints
from valens.stream import OutputStream, gen_addr_ipc
from valens.nodes import *

import cv2
import h5py
import numpy as np
import os
import pytest

def test_pose_topology_fullset():
    topology = va.pose.topology()
    assert topology == [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [17, 0], [17, 5], [17, 6], [17, 11], [17, 12]]

def test_pose_topology_subset():
    keypoints = [Keypoints.RHIP, Keypoints.RKNEE, Keypoints.RANKLE, Keypoints.NECK]
    topology = va.pose.topology(keypoints)
    assert topology == [[16, 14], [14, 12], [17, 12]]

def test_pose_filter_keypoints():
    keypoints = [Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK]

    full_pose = np.empty((len(Keypoints), 2))
    mask = np.zeros((len(Keypoints),), dtype=np.bool)
    for keypoint in keypoints:
        mask[keypoint.value] = 1
    pose = np.array([
        [0.5, 0.35], # hip
        [0.5, 0.5], # knee
        [0.5, 0.7], # ankle 
        [0.5, 0.1] # neck
    ])
    full_pose[mask] = pose
    
    filtered = va.pose.filter_keypoints(full_pose, keypoints)
    np.testing.assert_equal(pose[0, :], filtered[2, :])
    np.testing.assert_equal(pose[1, :], filtered[1, :])
    np.testing.assert_equal(pose[2, :], filtered[0, :])
    np.testing.assert_equal(pose[3, :], filtered[3, :])

def test_pose_trt_to_image():
    with h5py.File(va.constants.TEST_DATA_DIR + "/structures_pose_trt_to_image.h5", "r") as data:
        counts = data["counts"][:]
        objects = data["objects"][:]
        peaks = data["peaks"][:]
        expected_frame = data["frame"][:]
    
    topology = va.pose.topology()

    p = va.pose.nn_to_numpy(counts, objects, peaks)

    capture = cv2.VideoCapture(va.constants.TEST_DATA_DIR + "/BS_good_Jan15_1_20.mp4")
    _, actual_frame = capture.read()
    va.pose.draw_on_image(p, actual_frame, topology)

    np.testing.assert_equal(actual_frame, expected_frame)

def test_pose_multiple_persons():
    object_counts = []
    objects = []
    normalized_peaks = []
    for i in range(2):
        with h5py.File(va.constants.TEST_DATA_DIR + "/multiperson_nn_frame_" + str(i) + ".h5", "r") as data:
            object_counts.append(data['object_counts'][:])
            objects.append(data['objects'][:])
            normalized_peaks.append(data['normalized_peaks'][:])

    prev = None
    for i in range(2):
        pose = va.pose.nn_to_numpy(object_counts[i], objects[i], normalized_peaks[i], prev=prev)
        prev = pose

# def test_pose_video_sink():
#     with h5py.File(va.constants.TEST_DATA_DIR + "/structures_pose_trt_to_image.h5", "r") as data:
#         counts = data["counts"][:]
#         objects = data["objects"][:]
#         peaks = data["peaks"][:]
#         p = va.pose.nn_to_numpy(counts, objects, peaks)
    
#     capture = cv2.VideoCapture(va.constants.TEST_DATA_DIR + "/BS_good_Jan15_1_20.mp4")
#     _, frame = capture.read()

#     total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
#     print("Total frames:", total_frames)
    
#     frame_addr = gen_addr_ipc("frame")
#     pose_addr = gen_addr_ipc("pose")

#     frame_stream = OutputStream(frame_addr)
#     pose_stream = OutputStream(pose_addr)
#     sink = LocalDisplaySink(frame_addr, pose_addr, exercise_type='BS')
#     sink.start()

#     [s.start() for s in [frame_stream, pose_stream]]
#     frame_stream.send(frame)
#     pose_stream.send(p)

#     frame_stream.send(None)
#     pose_stream.send(None)
#     sink.join()

def test_pose_h5_sink():
    with h5py.File(va.constants.TEST_DATA_DIR + "/structures_pose_trt_to_image.h5", "r") as data:
        counts = data["counts"][:]
        objects = data["objects"][:]
        peaks = data["peaks"][:]
        p = va.pose.nn_to_numpy(counts, objects, peaks)

    expected_filename = va.constants.TEST_DATA_DIR + "/BS_good_Jan15_1_20.h5"
    actual_filename = va.constants.TEST_DATA_DIR + "/tmp_pose_sink.h5"

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
    pose = np.array([
        [0.5, 0.7],
        [0.5, 0.5],
        [0.5, 0.35],
        [0.5, 0.1]
    ])

    vectors = va.pose.vectors(pose)

    assert vectors.shape == (3, 2)
    for i in range(3):
        np.testing.assert_equal(vectors[i, :], [0, -1])

def test_pose_angles():
    pose = np.array([
        [0.7, 0.7],
        [0.7, 0.5],
        [0.5, 0.5],
        [0.5, 0.1]
    ])

    angles = va.pose.angles(pose, space_frame=[0, 1])

    assert angles.shape == (3,)
    assert angles[0] == np.pi
    assert angles[1] == np.pi / 2
    assert angles[2] == np.pi / 2

def test_pose_transform_zero():
    zero_pose = np.array([
        [0.5, 0.7],
        [0.5, 0.5],
        [0.5, 0.35],
        [0.5, 0.1]
    ])

    num_joints = zero_pose.shape[0]
    L = [0.0]
    for i in range(1, num_joints):
        joint1 = i - 1
        joint2 = i

        dist = np.linalg.norm(zero_pose[joint2, :] - zero_pose[joint1, :])
        L.append(dist + L[i - 1])

    Slist = np.empty((6, num_joints))
    for i in range(num_joints):
        Slist[:, i] = [0, 0, 1, -L[i], 0, 0]

    joint_angles = np.zeros((4,))
    transformed_pose = va.pose.transform(zero_pose, Slist, joint_angles)
    np.testing.assert_allclose(transformed_pose, zero_pose)

def test_pose_transform_nonzero():
    zero_pose = np.array([
        [0.5, 0.7],
        [0.5, 0.5],
        [0.5, 0.35],
        [0.5, 0.1]
    ])

    num_joints = zero_pose.shape[0]
    L = [0.0]
    for i in range(1, num_joints):
        joint1 = i - 1
        joint2 = i

        dist = np.linalg.norm(zero_pose[joint2, :] - zero_pose[joint1, :])
        L.append(dist + L[i - 1])

    Slist = np.empty((6, num_joints))
    for i in range(num_joints):
        Slist[:, i] = [0, 0, 1, -L[i], 0, 0]

    expected_pose = zero_pose.copy()
    expected_pose[2, 0] -= abs(zero_pose[2, 1] - zero_pose[1, 1])
    expected_pose[2, 1] = zero_pose[1, 1]
    expected_pose[3, 1] += abs(zero_pose[2, 1] - zero_pose[1, 1])
    expected_pose[3, 0] = expected_pose[2, 0]

    angles = va.pose.angles(expected_pose, space_frame=[0, 1])
    
    joint_angles = np.array([np.pi - angles[0], -angles[1], angles[2], 0])
    transformed_pose = va.pose.transform(zero_pose, Slist, joint_angles)

    np.testing.assert_equal(transformed_pose, expected_pose)

def test_pose_transform_link_length():
    zero_pose = np.array([
        [0.5, 0.7],
        [0.5, 0.5],
        [0.5, 0.35],
        [0.5, 0.1]
    ])

    num_joints = zero_pose.shape[0]
    L = [0.0]
    for i in range(1, num_joints):
        joint1 = i - 1
        joint2 = i

        dist = np.linalg.norm(zero_pose[joint2, :] - zero_pose[joint1, :])
        L.append(dist + L[i - 1])

    Slist = np.empty((6, num_joints))
    for i in range(num_joints):
        Slist[:, i] = [0, 0, 1, -L[i], 0, 0]

    angles = np.array([2.80488046, 1.31963791, 1.27038786]) 

    joint_angles = np.array([np.pi - angles[0], -angles[1], angles[2], 0])
    transformed_pose = va.pose.transform(zero_pose, Slist, joint_angles)

    for i in range(1, joint_angles.shape[0]):
        joint1 = i - 1
        joint2 = i

        dist1 = np.linalg.norm(zero_pose[joint2, :] - zero_pose[joint1, :])
        dist2 = np.linalg.norm(transformed_pose[joint2, :] - transformed_pose[joint1, :])
        
        np.testing.assert_allclose(dist1, dist2)
