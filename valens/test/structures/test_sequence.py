from valens import constants
from valens.structures import sequence

import h5py
import cv2
import numpy as np
from scipy import ndimage

def test_seq_visualize_with_input():
    filename = constants.DATA_DIR + "/sequences/BS_bad_1.h5"

    with h5py.File(filename, "r+") as data:
        seq = data["pose"][:]
    sequence.clean_nans(seq)
    # sequence.center(reps[1])
    colors = [(255, 255, 255)]
    filename = constants.DATA_DIR + "/outputs/BS_bad_1.mp4"
    capture = cv2.VideoCapture(constants.DATA_DIR + "/recordings/BS_bad_1.mp4")
    mask = sequence.gen_keypoints_mask(["neck", "right_hip", "right_knee", "right_ankle"])
    seq[~mask] = np.nan
    sequence.visualize([seq], colors, filename, capture=capture, fps=30)

def test_seq_visualize_rep():
    filename = constants.DATA_DIR + "/sequences/BS_bad_1.h5"

    with h5py.File(filename, "r+") as data:
        # data["reps"][:] = []
        # reps = data["reps"][:].tolist()
        p = data["pose"][:]

    rep_idx = sequence.get_rep_idx([80])
    reps = sequence.get_reps(p, rep_idx)
    # sequence.center(reps[0])
    # sequence.center(reps[1])
    # print(reps[0])
    colors = [(255, 255, 255)]

    seq = reps[0]
    sequence.clean_nans(seq)
    # for k in range(seq.shape[0]):
    #     for i in range(2):
    #         seq[k, i, :] = ndimage.median_filter(seq[k, i, :], 5)
    # mask = sequence.gen_keypoints_mask(["right_hip", "left_hip", "right_knee", "left_knee", "right_ankle", "left_ankle", "neck"])
    mask = sequence.gen_keypoints_mask(["neck", "right_hip", "right_knee", "right_ankle"])
    seq[~mask] = np.nan

    sequence.visualize([seq], colors, constants.DATA_DIR + "/outputs/BS_bad_1_r1.mp4", fps=30)

def test_seq_dtw():
    filename = constants.DATA_DIR + "/sequences/BS_good_1.h5"

    with h5py.File(filename, "r+") as data:
        # data["reps"][:] = [16, 36]
        # reps = data["reps"][:].tolist()
        p = data["pose"][:]

    rep_idx = sequence.get_rep_idx([80, 180])
    reps = sequence.get_reps(p, rep_idx)

    seq2 = p[:, :, rep_idx[0, 0]:rep_idx[0, 1]]
    # seq2 = p[:, :, rep_idx[1, 0]:rep_idx[1, 1]]

    filename = constants.DATA_DIR + "/sequences/BS_good_1.h5"

    with h5py.File(filename, "r+") as data:
        # data["reps"][:] = [16, 36]
        # reps = data["reps"][:].tolist()
        p = data["pose"][:]

    rep_idx = sequence.get_rep_idx([80, 180])
    reps = sequence.get_reps(p, rep_idx)

    #seq1 = p[:, :, rep_idx[0, 0]:rep_idx[0, 1]]
    seq1 = p[:, :, rep_idx[1, 0]:rep_idx[1, 1]]

    sequence.clean_nans(seq1)
    # sequence.clean_nans(seq2)

    # keypoints = ["right_hip"]
    # seq1 = sequence.filter_keypoints(seq1, keypoints)
    # seq2 = sequence.filter_keypoints(seq2, keypoints)

    # # pose 1 is the same in seq1 and seq2
    first = seq1[:, :, 0]
    second = seq2[:, :, 0]
    neck = sequence.all_keypoints.index("neck")
    ankle = sequence.all_keypoints.index("right_ankle")
    h1 = first[neck, 1] - first[ankle, 1]
    h2 = second[neck, 1] - second[ankle, 1]
    r = h1 / h2
    print(h1, h2, r)
    seq2[:, 1, :] *= r
    sequence.center(seq1)
    sequence.center(seq2)


    #for k in range(seq1.shape[0]):
    #    for i in range(2):
    #        diff = first[k, i] - second[k, i]
    #        seq2[k, i, :] += diff

    alignments, cost = sequence.warp(seq1, seq2)
    aligned_seq2 = sequence.align(seq1, seq2, alignments)
    mask = sequence.gen_keypoints_mask(["neck", "right_hip", "right_knee", "right_ankle"])
    seq1[~mask] = np.nan
    aligned_seq2[~mask] = np.nan
    print(cost[mask])

    colors = [(255, 255, 255), (0, 255, 0)]
    sequence.visualize([seq1, aligned_seq2], colors, constants.DATA_DIR + "/outputs/BS_good_1_r2_dtw.mp4", fps=30)
