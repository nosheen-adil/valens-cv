from valens import constants
from valens.structures import sequence, exercise
from valens.structures.exercise import Keypoints

import argparse
import h5py
import json
import numpy as np
import os
import random
import modern_robotics as mr

def transform(initial_pose, joint_angles, topology, joints):
    '''
    pose: hip, knee, ankle, neck
    joint_angles: ankle, knee, hip, neck [2, 1, 0, 3]
    topology: [knee, ankle], [hip, knee], [neck, hip]
    joints: ankle, knee, hip, neck
    '''
    num_links = len(topology)
    num_joints = num_links + 1

    # calculate link lengths based on pos of joints in pose
    L = [0.0]
    for i in range(1, num_joints):
        joint1 = topology[i-1][0]
        joint2 = topology[i-1][1]

        dist = np.linalg.norm(initial_pose[joint1, :] - initial_pose[joint2, :])
        L.append(dist + L[i - 1])

    pose = np.array([
        [initial_pose[2, 0], initial_pose[2, 1] - L[2]],
        [initial_pose[2, 0], initial_pose[2, 1] - L[1]],
        [initial_pose[2, 0], initial_pose[2, 1] - L[0]], # ankle
        [initial_pose[2, 0], initial_pose[2, 1] - L[3]],
    ])

    pose_space = pose.copy()
    pose_space = 1.0 - pose_space
    space_frame = pose_space[2, :].copy()

    pose_space[:, 0] -= space_frame[0]
    pose_space[:, 1] -= space_frame[1]

    # create screw axis for each joint
    # ankle, knee, hip, neck
    Slist = np.empty((6, num_joints))
    for i in range(num_joints):
        Slist[:, i] = [0, 0, 1, L[i], 0, 0]

    corrected_pose = pose.copy()
    for i, joint in enumerate(joints):
        # calculate zero frame of each joint (excluding the ankle) from pose
        M = np.array([
            [1, 0, 0, pose_space[joint, 0]],
            [0, 1, 0, pose_space[joint, 1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # calculate iterative PoE and apply to each joint (excluding the ankle)
        T = mr.FKinSpace(M, Slist[:, 0:i+1], joint_angles[0:i+1])

        # extract pos from each transformation matrix and return the transformed pose
        pos = T[0:2, 3]
        corrected_pose[joint, :] = pos

    corrected_pose[:, 0] += space_frame[0]
    corrected_pose[:, 1] += space_frame[1]
    corrected_pose = 1 - corrected_pose
    return corrected_pose

def calc_body_vecs(pose, topology):
    p = 1 - pose
    p[:, 0] -= p[2, 0]
    p[:, 1] -= p[2, 1]

    num_vecs = len(topology)
    body_vecs = np.empty((num_vecs + 1, 2))

    for v in range(num_vecs):
        for i in range(2):
            body_vecs[v, i] =  np.array([p[topology[v][0], i] - p[topology[v][1], i]])
        body_vecs[v] /= np.linalg.norm(body_vecs[v])
    body_vecs[-1, :] = [0, 1]
    return body_vecs

def calc_body_angles(body_vecs, connections=[]):
    num_vecs = body_vecs.shape[0]
    if not connections:
        connections = [[i, i+1] for i in range(num_vecs - 1)]
    body_angles = np.empty((len(connections),))
    for c in range(len(connections)):
        joint1 = connections[c][0]
        joint2 = connections[c][1]
        body_angles[c] = np.arccos(body_vecs[joint1, 0] * body_vecs[joint2, 0] + body_vecs[joint1, 1] * body_vecs[joint2, 1])
    return body_angles

def load_features(seq):
    hip = 0
    knee = 1
    ankle = 2
    neck = 3

    x = 0
    y = 1

    T = seq.shape[-1]
    topology = [[neck, hip], [hip, knee], [knee, ankle]]
    body_angles = np.empty((T,3))
    for t in range(T):
        body_vecs = calc_body_vecs(seq[:, :, t], topology)
        body_angles[t, :] = calc_body_angles(body_vecs, connections=[[0, 1], [1, 2], [2, 3]])
    
    return body_angles

def gen_seq(reps):
    # reps = [reps[0]]
    initial_pose = np.array([
        [0.5, 0.35],
        [0.5, 0.5],
        [0.5, 0.7],
        [0.5, 0.1]
    ])
    hip = 0
    knee = 1
    ankle = 2
    neck = 3

    x = 0
    y = 1

    mask = exercise.BodyweightSquat().keypoint_mask()

    topology = [[neck, hip], [hip, knee], [knee, ankle]]
    for i in range(len(reps)):
        T = reps[i].shape[-1]
        # new_seq = np.empty((4, 2, T))
        seq = reps[i][mask].copy()
        sequence.clean_nans(seq)
        sequence.filter_noise(seq)
        for t in range(T):
            body_vecs = calc_body_vecs(seq[:, :, t], topology)
            body_angles = calc_body_angles(body_vecs, connections=[[0, 1], [1, 2], [2, 3]])
            joint_angles = np.array([body_angles[2], -body_angles[1], body_angles[0], 0.0])
            # joint_angles = np.array([0.0, 0.0, 0.0, 0.0])
            pose = transform(initial_pose, joint_angles, topology[::-1], joints=[2, 1, 0, 3])
            reps[i][mask, :, t] = pose

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
    e = exercise.load(exercise_type)
    color = (255, 255, 255)

    # _, _, input_filenames = next(os.walk(args.sequences_dir))
    input_filenames = ['BS_bad_1.h5', 'BS_good_1.h5']
    input_filenames = [args.sequences_dir + '/' + input_filename for input_filename in input_filenames if input_filename[0:len(exercise_type)] == exercise_type]
    
    rep_idxx = [[90, 180, 270], [90, 180]]
    reps = []
    for i, input_filename in enumerate(input_filenames):
        with h5py.File(input_filename, 'r') as data:
            seq = data["pose"][:]

        # sequence.clean_nans(seq)
        # sequence.filter_noise(seq)

        rep_idx = sequence.get_rep_idx(rep_idxx[i])
        print(rep_idx)
        # rep_idx = e.detect_reps(seq)

        reps += sequence.get_reps(seq, rep_idx)
        # e.detect_side(reps[0])
        # keypoint_mask = e.keypoint_mask()

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
