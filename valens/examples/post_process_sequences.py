from valens import constants
from valens.structures import sequence, exercise

import argparse
import h5py
import json
import numpy as np
import os
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a reference sequence from a pose sequence using DTW')
    parser.add_argument('--exercise', default='', help='Short-form of exercise')
    parser.add_argument('--sequences_dir', default=constants.DATA_DIR + '/sequences', help='Directory to store reference sequence')
    parser.add_argument('--outputs_dir', default=constants.DATA_DIR + '/outputs', help='Directory to store output videos')
    parser.add_argument('--fps', default=30, help='Fps of output video')

    args = parser.parse_args()
    
    exercise_type = args.exercise
    e = exercise.load(exercise_type)
    color = (255, 255, 255)

    _, _, input_filenames = next(os.walk(args.sequences_dir))
    input_filenames = [args.sequences_dir + '/' + input_filename for input_filename in input_filenames if input_filename[0:len(exercise_type)] == exercise_type]
    for input_filename in input_filenames:
        with h5py.File(input_filename, 'r') as data:
            seq = data["pose"][:]

        sequence.clean_nans(seq)
        sequence.filter_noise(seq)
        rep_idx = e.detect_reps(seq)
        reps = sequence.get_reps(seq, rep_idx)
        e.detect_side(reps[0])
        keypoint_mask = e.keypoint_mask()
        
        name = os.path.splitext(os.path.basename(input_filename))[0]
        label = name.find('good') == 3
        print(label, name)
        # exit(0)
        for i in range(len(reps)):
            seq = reps[i][keypoint_mask]
            e.center(seq)

            output_filename = args.sequences_dir + '/post_processed/' + name + '_r' + str(i) + '.h5'
            print(output_filename)
            with h5py.File(output_filename, 'w') as data:
                data['pose'] = seq
                data['label'] = label
                
            output_filename = args.outputs_dir + '/post_processed/' + name + '_r' + str(i) + '.mp4'
            print(output_filename)
            sequence.visualize([seq], [color], output_filename, e.topology(), fps=int(args.fps))
