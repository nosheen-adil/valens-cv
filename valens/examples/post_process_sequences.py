from valens import constants
from valens import structures as core
import valens.structures.exercise

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
    exercise = core.exercise.load(exercise_type)
    
    _, _, input_filenames = next(os.walk(args.sequences_dir))
    input_filenames = [args.sequences_dir + '/' + input_filename for input_filename in input_filenames if input_filename[0:len(exercise_type)] == exercise_type]
    
    for input_filename in input_filenames:
        with h5py.File(input_filename, 'r') as data:
            seq = data["pose"][:]
        
        name = os.path.splitext(os.path.basename(input_filename))[0]
        label = name.find('good') == 3
        print(label, name)
        
        started = False
        total_reps = 0
        for t in range(seq.shape[-1]):
            pose = seq[:, :, t]
            exercise.fit(pose)
            if exercise.in_rep and not started:
                print('started', t)
                started = True
            elif not exercise.in_rep and started:
                print('finished', t)
                started = False
                rep = exercise.prev_rep

                output_filename = args.sequences_dir + '/post_processed/' + name + '_r' + str(total_reps) + '.h5'
                print(rep.shape, output_filename)
                with h5py.File(output_filename, 'w') as data:
                    data['seq'] = rep
                    data['label'] = label

                total_reps += 1
