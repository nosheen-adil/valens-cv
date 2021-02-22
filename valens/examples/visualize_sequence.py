from valens import constants
from valens.structures import sequence, exercise

import argparse
import h5py
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualizes a pose sequence, optionally overlayed on an input video')
    parser.add_argument('--name', default='', help='Name of input video')
    parser.add_argument('--fps', default=30, help='Fps of ouput videos')
    parser.add_argument('--recordings_dir', default=constants.DATA_DIR + '/recordings', help='Directory for recordings')
    parser.add_argument('--sequences_dir', default=constants.DATA_DIR + '/sequences', help='Directory for sequences')
    parser.add_argument('--outputs_dir', default=constants.DATA_DIR + '/outputs', help='Directory to store output videos')

    args = parser.parse_args()

    # input_url = args.recordings_dir + '/' + args.name + '.mp4'
    input_filename = args.sequences_dir + '/' + args.name + '.h5'
    with h5py.File(input_filename, 'r+') as data:
        # del data["reps"]
        # data["reps"] = [90, 180]
        seq = data["pose"][:]
        # reps = data["reps"][:].tolist()
    
    # rep_idx = sequence.get_rep_idx(reps)
    sequence.clean_nans(seq)
    sequence.filter_noise(seq)

    rep_idx = sequence.detect_reps(seq, drop_percent=0.8)
    reps = sequence.get_reps(seq, rep_idx)
    color = (255, 255, 255)

    e = exercise.load(args.name[0:2])
    # e.side = exercise.Side.LEFT
    keypoint_mask = e.keypoint_mask()

    for i in range(len(reps)):
        seq = reps[i][keypoint_mask]

        output_filename = args.outputs_dir + '/' + args.name + '_r' + str(i) + '.mp4'
        print(output_filename)
        sequence.visualize([seq], [color], output_filename, e.topology(), fps=int(args.fps))
