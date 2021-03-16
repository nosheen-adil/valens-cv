ROOT_DIR = "/valens-cv"
PYTHON_DIR = ROOT_DIR + "/valens"
DATA_DIR = PYTHON_DIR + "/data"

TEST_DATA_DIR = PYTHON_DIR + "/test/data"

POSE_MODEL_WIDTH = 224
POSE_MODEL_HEIGHT = 224
POSE_MODEL_TRT_WEIGHTS = DATA_DIR + "/resnet18_baseline_att_224x224_A_epoch_249_trt.pth"
POSE_JSON = DATA_DIR + "/human_pose.json"

AWS_ENDPOINT_JSON = DATA_DIR + '/aws/endpoint.json'
AWS_CERT_PATH = DATA_DIR + '/aws/device.pem.crt'
AWS_KEY_PATH = DATA_DIR + '/aws/private.pem.key'
AWS_ROOT_CA_PATH = DATA_DIR + '/aws/Amazon-root-CA-1.pem'
