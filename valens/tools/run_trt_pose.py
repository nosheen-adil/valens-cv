import cv2
import h5py
import json
import PIL.Image
import numpy as np
import time
import torch
from torch2trt import TRTModule
import torchvision
import torchvision.transforms as transforms
import trt_pose.coco
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

cv2.namedWindow('image',cv2.WINDOW_NORMAL)

WIDTH = 224
HEIGHT = 224
DATA_PATH = "/valens-cv/valens/data"

with open(DATA_PATH + '/human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

OPTIMIZED_MODEL = DATA_PATH + '/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
print("Loaded optimized model")

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

def execute(image, writer=None):
    start = time.time()
    data = preprocess(image)

    cmap, paf = model_trt(data)
    # end = time.time()
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    end = time.time()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    print("counts", counts[0])
    # end = time.time()
    if counts[0] > 0:
        counts[0] = 1
    draw_objects(image, counts, objects, peaks)
    # end = time.time()
    # writer.write(image)
    # img = cv2.imencode('.jpg', image[:, ::-1, :])[1]
    # cv2.imshow('image', image[:, ::-1, :])
    # cv2.waitKey(1)
    # end = time.time()
    print("Fps:", 1 / (end - start))
    with h5py.File("/valens-cv/valens/test/data/structures_pose_trt_to_image.h5", "w") as data:
        data["counts"] = counts.numpy()
        data["objects"] = objects.numpy()
        data["peaks"] = peaks.numpy()
        data["frame"] = image
        exit(0)

capture = cv2.VideoCapture(DATA_PATH + "/recordings/good_BS_Jan15_1_20.mp4")
# capture = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# writer = cv2.VideoWriter(DATA_PATH + '/outputs/good_BS_Jan15_1_20.mp4', fourcc, capture.get(cv2.CAP_PROP_FPS), (224, 224))
while True:
    ret, frame = capture.read()
    if not ret:
        break
    frame = cv2.resize(frame, dsize=(WIDTH, HEIGHT))
    # execute(frame, writer)
    execute(frame)
