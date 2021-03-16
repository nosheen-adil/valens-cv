from valens import constants
import valens as va
from valens import sequence, exercise, pose
from valens.exercise import Keypoints

import argparse
import h5py
import json
import numpy as np
import os
import random

def get_rep_idx(markers):
    rep_idx = np.empty((len(markers) + 1,2), dtype=np.int32)
    markers.insert(0, 0.0)
    markers.append(-1)
    for i in range(len(markers) - 1):
        rep_idx[i, :] = [markers[i], markers[i+1]]
    return rep_idx

def get_reps(p, rep_idx):
    refs = []
    for i in range(rep_idx.shape[0]):
        refs.append(p[:, :, rep_idx[i, 0]:rep_idx[i, 1]])
    return refs

def gen_seq(reps):
    initial_pose = np.array([
        [0.5, 0.7],
        [0.5, 0.5],
        [0.5, 0.35],
        [0.5, 0.1]
    ])
    keypoints = [Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK]
    topology = [[0, 1], [1, 2], [2, 3]] # [ankle -> knee, knee -> hip, hip -> neck]

    for i in range(len(reps)):
        T = reps[i].shape[-1]
        seq = reps[i].copy()
        for t in range(T):
            pose = va.pose.filter_keypoints(seq[:, :, t], keypoints)
            angles = va.pose.angles(pose, topology)
            joint_angles = np.array([np.pi - angles[0], -angles[1], angles[2], 0.0])
            # joint_angles = np.array([0.0, 0.0, 0.0, 0.0])
            pose = va.pose.transform(initial_pose, joint_angles, topology)
            for k in range(pose.shape[0]):
                reps[i][keypoints[k].value, :, t] = pose[k, :]

    # random.shuffle(reps)
    total_seq = None
    for rep in reps:
        if total_seq is None:
            total_seq = rep
        else:
            total_seq = np.concatenate((total_seq, rep), axis=2)
    return total_seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a reference sequence from a pose sequence using DTW')
    # parser.add_argument('--name', default='', help='Name of file')
    parser.add_argument('--sequences_dir', default=constants.DATA_DIR + '/sequences', help='Directory to store reference sequence')
    parser.add_argument('--exercise', default='', help='Short-form of exercise')

    args = parser.parse_args()
    
    exercise_type = args.exercise

    # _, _, input_filenames = next(os.walk(args.sequences_dir))
    input_filenames = ['BS_bad_1.h5', 'BS_good_1.h5']
    input_filenames = [args.sequences_dir + '/' + input_filename for input_filename in input_filenames if input_filename[0:len(exercise_type)] == exercise_type]
    
    rep_idxx = [[90, 180, 270], [90, 180]]
    reps = []
    for i, input_filename in enumerate(input_filenames):
        with h5py.File(input_filename, 'r') as data:
            seq = data["pose"][:]

        rep_idx = get_rep_idx(rep_idxx[i])
        print(rep_idx)

        reps += get_reps(seq, rep_idx)

    print(len(reps))
    total_seq = gen_seq(reps)
    print(total_seq.shape)

    # for i in range(1):
    #     seq = seq[:, :, 0:90]
    #     # e.center(seq)
    #     seq = np.tile(seq, (1, 1, n))
    #     print(seq.shape)

    output_filename = args.sequences_dir + '/BS_good1xbad1.h5'
    print(output_filename)
    with h5py.File(output_filename, 'w') as data:
        data['pose'] = total_seq
        # data['label'] = label

    
        # n = 10
        # for i in range(1):
        #     seq = seq[:, :, 0:90]
        #     # e.center(seq)
        #     seq = np.tile(seq, (1, 1, n))
        #     print(seq.shape)

        #     output_filename = args.sequences_dir + '/' + name + '_repeatedx' + str(n) + '.h5'
        #     print(output_filename)
        #     with h5py.File(output_filename, 'w') as data:
        #         data['pose'] = seq
        #         data['label'] = label
                
            # output_filename = args.outputs_dir + '/post_processed/' + name + '_r' + str(i) + '.mp4'
            # print(output_filename)
            # sequence.visualize([seq], [color], output_filename, e.topology(), fps=int(args.fps))
