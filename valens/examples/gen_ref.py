from valens import constants
from valens.structures import sequence, exercise

import argparse
import h5py
import json
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a reference sequence from a pose sequence using DTW')
    parser.add_argument('--name', default='', help='Name of .h5 file with sequences')
    parser.add_argument('--sequences_dir', default=constants.DATA_DIR + '/sequences', help='Directory to store reference sequence')
    parser.add_argument('--outputs_dir', default=constants.DATA_DIR + '/outputs', help='Directory to store output videos')
    parser.add_argument('--fps', default=30, help='Fps of output video')

    args = parser.parse_args()
    
    reps = []
    for name in args.name.split(','):
        input_filename = args.sequences_dir + '/' + name + '.h5'
        with h5py.File(input_filename, 'r') as data:
            seq = data["pose"][:]
            # r = data["reps"][:].tolist()
        
        sequence.clean_nans(seq)
        sequence.filter_noise(seq)
        # rep_idx = sequence.get_rep_idx(r)
        rep_idx = sequence.detect_reps(seq, drop_percent=0.8)
        reps += sequence.get_reps(seq, rep_idx)
    
    exercise_type = args.name[0:2]
    e = exercise.load(exercise_type)
    keypoint_mask = e.keypoint_mask() 

    print(len(reps))

    merged_seq = reps[0][keypoint_mask]
    
    # sequence.filter_noise(merged_seq)
    # e.center(merged_seq)
    # merged_seq = None
    # for i in range(1, len(reps) - 1):
    #     if i == 1:
    #         seq1 = reps[i - 1][keypoint_mask]
    #     else:
    #         seq1 = merged_seq
    #     seq2 = reps[i][keypoint_mask]

    #     sequence.clean_nans(seq1)
    #     sequence.clean_nans(seq2)

    #     e.normalize(seq1, seq2, axis=1)
    #     e.center(seq1)
    #     e.center(seq2)

    #     alignments, cost = sequence.warp(seq1, seq2)
    #     aligned_seq2 = sequence.align(seq1, seq2, alignments)
    #     merged_seq = sequence.merge(seq1, aligned_seq2)
    
    output_filename = args.outputs_dir + '/' + exercise_type + '_ref.mp4'
    sequence.visualize([merged_seq], [(255, 255, 255)], output_filename, e.topology(), fps=int(args.fps))

    output_filename = args.sequences_dir + '/' + exercise_type + '_ref.h5'
    with h5py.File(output_filename, 'w') as data:
        data['pose'] = merged_seq
