import json
import trt_pose.coco
from torch2trt import TRTModule
import torch

import time

import cv2
import torchvision.transforms as transforms
import PIL.Image

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

from jetcam.usb_camera import USBCamera
# from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg

#import ipywidgets
#from IPython.display import display
import numpy as np
#image_w = ipywidgets.Image(format='jpeg')

#display(image_w)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)

WIDTH = 224
HEIGHT = 224

with open('data/human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

OPTIMIZED_MODEL = 'data/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
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

def execute(change):
    start = time.time()
    image = change['new']
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    #image_w.value = bgr8_to_jpeg(image[:, ::-1, :])
    img = cv2.imencode('.jpg', image[:, ::-1, :])[1]
#np.frombuffer(bgr8_to_jpeg(image[:, ::-1, :]))
    print(type(image))
    cv2.imshow('image', image[:, ::-1, :])
    cv2.waitKey(1)
    end = time.time()
    print("fps:", 1 / (end - start))

camera = USBCamera(width=WIDTH, height=HEIGHT, capture_fps=30)
# camera = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=30)

camera.running = True

camera.observe(execute, names='value')
