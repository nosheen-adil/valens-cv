from valens import constants
from valens import structures as core
from valens.structures.pose import Keypoints

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def test_sequence_dtw_knn():
    keypoints = [Keypoints.RANKLE, Keypoints.RKNEE, Keypoints.RHIP, Keypoints.NECK]
    topology = [[0, 1], [1, 2], [2, 3]] # [ankle -> knee, knee -> hip, hip -> neck]
    filenames = core.sequence.post_processed_filenames(exercise_type='BS')
    # print(filenames)

    X_train_names, X_test_names = train_test_split(filenames, test_size=0.4, random_state=42)
    print(len(X_train_names))

    X_train, y_train = core.sequence.load_features(X_train_names)
    X_test, y_test = core.sequence.load_features(X_test_names)

    classifier = core.sequence.DtwKnn()
    classifier.fit(X_train, y_train)
    predictions = []
    for test in range(len(X_test_names)):
        label, _ = classifier.predict(X_test[test])
        predictions.append(label)

    print(classification_report(y_test, predictions, target_names=['correct', 'bad']))
