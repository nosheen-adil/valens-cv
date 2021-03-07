from valens import constants
from valens import structures as core

import h5py
import numpy as np

def test_exercise_bs_add():
    exercise = core.exercise.load('BS')
    
    with h5py.File(constants.TEST_DATA_DIR + '/BS_good_1.h5', 'r') as data:
        seq = data['pose'][:]

    T = seq.shape[-1]
    started = False
    reps = []
    for t in range(T):
        pose = seq[:, :, t]
        exercise.fit(pose)
        if exercise.in_rep and not started:
            started =  True
            reps.append([t])
        if not exercise.in_rep and started:
            started = False
            reps[-1].append(t)
    np.testing.assert_equal(reps, [[31, 48], [124, 152], [230, 259]])
