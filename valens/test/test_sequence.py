from valens import constants
import valens as va
from valens.pose import Keypoints

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

def test_sequence_bs_classify():
    filenames = va.sequence.post_processed_filenames(exercise_type='BS')
    # print(filenames)

    X_train_names, X_test_names = train_test_split(filenames, test_size=0.4, random_state=42)
    print(len(X_train_names))

    X_train, y_train = va.sequence.load_features(X_train_names)
    X_test, y_test = va.sequence.load_features(X_test_names)

    classifier = va.sequence.DtwKnn()
    classifier.fit(X_train, y_train)
    predictions = []
    for test in range(len(X_test_names)):
        label, _ = classifier.predict(X_test[test])
        predictions.append(label)

    print(classification_report(y_test, predictions, target_names=['correct', 'bad']))

def test_sequence_bc_classify():
    filenames = va.sequence.post_processed_filenames(exercise_type='BC')
    # print(filenames)

    X_train_names, X_test_names = train_test_split(filenames, test_size=0.4, random_state=42)
    print(len(X_train_names), len(X_test_names))

    X_train, y_train = va.sequence.load_features(X_train_names)
    X_test, y_test = va.sequence.load_features(X_test_names)
    # for t in range(len(X_train)):
    #     X_train[t] = np.array([X_train[t][0, :]])
    # for t in range(len(X_test)):
    #     X_test[t] = np.array([X_test[t][0, :]])

    classifier = va.sequence.DtwKnn()
    classifier.fit(X_train, y_train)
    predictions = []
    for test in range(len(X_test_names)):
        label, _ = classifier.predict(X_test[test])
        predictions.append(label)

    print(classification_report(y_test, predictions, target_names=['correct', 'bad']))

def test_sequence_match_frame():
    x = np.array([7.1, 7.0, 6.5, 5.3, 4.1, 3.2, 5.1, 6.6, 6.9, 7.1, 7.5])
    y = np.array([7.2, 6.5, 5.5, 5.4, 2.3, 2.1, 2.0, 1.9, 2.1, 2.7, 3.0, 4.4, 5.6, 6.9, 7.9])

    y_x = np.empty_like(y, dtype=np.uint32)
    j = 0
    for i in range(y.shape[0]):
        j = va.sequence.match_frame(x, j, y[i])
        y_x[i] = j

    expected = np.array([0, 2, 3, 3, 5, 5, 5, 5, 5, 5, 5, 6, 6, 8, 10], dtype=np.uint32)
    np.testing.assert_equal(y_x, expected)
    