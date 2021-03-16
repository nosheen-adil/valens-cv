import valens as va

import h5py
import numpy as np

# def test_exercise_bs_detect_side_right():
#     cases = ['BS_bad_1', 'BS_bad_2', 'BS_good_1', 'BS_good_2', 'BS_good_3', 'BS_good_4']
#     for case in cases:
#         print('case: ', case)
#         exercise = va.exercise.load('BS')
#         with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
#             seq = data['pose'][:]

#         T = seq.shape[-1]
#         print(T)
#         for t in range(T):
#             pose = seq[:, :, t]
#             exercise.fit(pose)
#         assert exercise.side is va.exercise.Side.RIGHT

# def test_exercise_bs_detect_side_left():
#     cases = ['BS_bad_3']
#     for case in cases:
#         print('case: ', case)
#         exercise = va.exercise.load('BS')
#         with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
#             seq = data['pose'][:]

#         T = seq.shape[-1]
#         print(T)
#         for t in range(T):
#             pose = seq[:, :, t]
#             exercise.fit(pose)
#         assert exercise.side is va.exercise.Side.LEFT

def test_exercise_bs_num_reps():
    cases = ['BS_good_1', 'BS_good_2', 'BS_good_3', 'BS_good_4', 'BS_bad_1', 'BS_bad_2', 'BS_bad_3']
    expected_num_reps = [3, 3, 2, 3, 4, 2, 5]

    for i, case in enumerate(cases):
        print('case', case)
        exercise = va.exercise.load('BS')
        with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
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
        print(reps)
        assert len(reps) == expected_num_reps[i]

def test_exercise_bs_correct_bad():
    cases = ['BS_bad_1', 'BS_bad_2', 'BS_bad_3']
    for case in cases:
        print('case: ', case)
        exercise = va.exercise.load('BS')
        with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
            seq = data['pose'][:]

        T = seq.shape[-1]
        print(T)
        false_count = 0
        for t in range(T):
            pose = seq[:, :, t]
            exercise.fit(pose)
            result = exercise.predict()
            if result is not None and 'feedback' in result:
                for feedback in result['feedback'].values():
                    if feedback['correct'] is False:
                        false_count += 1

        assert false_count > 0

def test_exercise_bs_correct_good():
    cases = ['BS_good_1', 'BS_good_2', 'BS_good_3', 'BS_good_4']
    for case in cases:
        print('case: ', case)
        exercise = va.exercise.load('BS')
        with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
            seq = data['pose'][:]

        T = seq.shape[-1]
        print(T)
        false_count = 0
        for t in range(T):
            pose = seq[:, :, t]
            exercise.fit(pose)
            result = exercise.predict()
            if result is not None and 'feedback' in result:
                for feedback in result['feedback'].values():
                    if feedback['correct'] is False:
                        false_count += 1

        assert false_count == 0

def test_exercise_bc_detect_side_left():
    cases = ['BC_good_1', 'BC_good_2', 'BC_bad_2', 'BC_bad_6']
    for case in cases:
        print('case: ', case)
        exercise = va.exercise.load('BC')
        with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
            seq = data['pose'][:]

        T = seq.shape[-1]
        print(T)
        for t in range(T):
            pose = seq[:, :, t]
            exercise.fit(pose)
        assert exercise.side is va.exercise.Side.LEFT
    
def test_exercise_bc_detect_side_right():
    cases = ['BC_good_3', 'BC_good_4', 'BC_good_5']
    expected_num_reps = [2, 2, 1, 1, 5, 5, 4, 5]
    for case in cases:
        print('case', case)
        exercise = va.exercise.load('BC')
        with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
            seq = data['pose'][:]

        T = seq.shape[-1]
        for t in range(T):
            pose = seq[:, :, t]
            exercise.fit(pose)
        assert exercise.side is va.exercise.Side.RIGHT

def test_exercise_bc_num_reps():
    cases = ['BC_good_1', 'BC_good_2', 'BC_good_3', 'BC_good_4', 'BC_good_5', 'BC_bad_2', 'BC_bad_6']
    expected_num_reps = [2, 2, 1, 1, 5, 4, 5]

    for i, case in enumerate(cases):
        print('case', case)
        exercise = va.exercise.load('BC')
        with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
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
        print(reps)
        assert len(reps) == expected_num_reps[i]

def test_exercise_bc_correct_bad():
    cases = ['BC_bad_1', 'BC_bad_2', 'BC_bad_6']
    for case in cases:
        print('case: ', case)
        exercise = va.exercise.load('BC')
        with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
            seq = data['pose'][:]

        T = seq.shape[-1]
        print(T)
        false_count = 0
        for t in range(T):
            pose = seq[:, :, t]
            exercise.fit(pose)
            result = exercise.predict()
            if result is not None and 'feedback' in result:
                for feedback in result['feedback'].values():
                    if feedback['correct'] is False:
                        false_count += 1

        assert false_count > 0

def test_exercise_bc_correct_good():
    cases = ['BC_good_1', 'BC_good_2', 'BC_good_3', 'BC_good_4', 'BC_good_5',]
    for case in cases:
        print('case: ', case)
        exercise = va.exercise.load('BC')
        with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
            seq = data['pose'][:]

        T = seq.shape[-1]
        print(T)
        false_count = 0
        for t in range(T):
            pose = seq[:, :, t]
            exercise.fit(pose)
            result = exercise.predict()
            if result is not None and 'feedback' in result:
                for feedback in result['feedback'].values():
                    if feedback['correct'] is False:
                        false_count += 1

        assert false_count == 0

def test_exercise_pu_detect_side_left():
    cases = ['PU_good_3', 'PU_bad_4']
    for case in cases:
        print('case: ', case)
        exercise = va.exercise.load('PU')
        with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
            seq = data['pose'][:]

        T = seq.shape[-1]
        print(T)
        for t in range(T):
            # print(t)
            pose = seq[:, :, t]
            exercise.fit(pose)
        assert exercise.side is va.exercise.Side.LEFT

def test_exercise_pu_detect_side_right():
    cases = ['PU_good_1', 'PU_good_2', 'PU_bad_1', 'PU_bad_2']
    for case in cases:
        print('case: ', case)
        exercise = va.exercise.load('PU')
        with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
            seq = data['pose'][:]

        T = seq.shape[-1]
        print(T)
        for t in range(T):
            pose = seq[:, :, t]
            exercise.fit(pose)
        assert exercise.side is va.exercise.Side.RIGHT

def test_exercise_pu_num_reps():
    cases = ['PU_good_1', 'PU_good_2', 'PU_good_3', 'PU_bad_1', 'PU_bad_2', 'PU_bad_4']
    expected_num_reps = [3, 2, 3, 3, 1, 1]

    for i, case in enumerate(cases):
        print('case', case)
        exercise = va.exercise.load('PU')
        with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
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
        print(reps)
        assert len(reps) == expected_num_reps[i]

def test_exercise_pu_correct_bad():
    cases = ['PU_bad_1', 'PU_bad_2', 'PU_bad_4']
    for case in cases:
        print('case: ', case)
        exercise = va.exercise.load('PU')
        with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
            seq = data['pose'][:]

        T = seq.shape[-1]
        print(T)
        false_count = 0
        for t in range(T):
            pose = seq[:, :, t]
            exercise.fit(pose)
            result = exercise.predict()
            if result is not None and 'feedback' in result:
                for feedback in result['feedback'].values():
                    if feedback['correct'] is False:
                        false_count += 1

        assert false_count > 0

def test_exercise_pu_correct_good():
    cases = ['PU_good_1', 'PU_good_2', 'PU_good_3']
    for case in cases:
        print('case: ', case)
        exercise = va.exercise.load('PU')
        with h5py.File(va.constants.DATA_DIR + '/sequences/' + case + '.h5', 'r') as data:
            seq = data['pose'][:]

        T = seq.shape[-1]
        print(T)
        false_count = 0
        for t in range(T):
            pose = seq[:, :, t]
            exercise.fit(pose)
            result = exercise.predict()
            if result is not None and 'feedback' in result:
                for feedback in result['feedback'].values():
                    if feedback['correct'] is False:
                        false_count += 1

        assert false_count == 0