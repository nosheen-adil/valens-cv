from valens import constants
from valens.node import Node
from valens import pose
from valens.stream import InputStream, OutputStream

import cv2
import json
import numpy as np
import PIL.Image
import torch
import torchvision.transforms as transforms
from torch2trt import TRTModule
import trt_pose.coco
from trt_pose.parse_objects import ParseObjects

class PoseFilter(Node):
    def __init__(self, frame_address, pose_address, model_path=constants.POSE_MODEL_TRT_WEIGHTS, pose_path=constants.POSE_JSON):
        super().__init__("PoseFilter")
        self.input_streams["frame"] = InputStream(frame_address)
        self.output_streams["pose"] = OutputStream(pose_address)

        self.model_path = model_path
        self.pose_path = pose_path
        self.prev = None

    def prepare(self):
        with open(self.pose_path, 'r') as f:
            human_pose = json.load(f)
        topology = trt_pose.coco.coco_category_to_topology(human_pose)
        self.parse_objects = ParseObjects(topology)

        self.model_trt = TRTModule()
        print(self.name + ": Loading model")
        self.model_trt.load_state_dict(torch.load(self.model_path))
        print(self.name + ": Loaded optimized model")

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

    def process(self):
        frame = self.input_streams["frame"].recv()
        if frame is None:
            self.stop()
            return

        data = self.preprocess(frame)
        cmap, paf = self.model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)
        p = pose.nn_to_numpy(counts, objects, peaks, prev=self.prev)
        self.prev = p.copy()
        self.output_streams["pose"].send(p)
        
    def preprocess(self, frame):
        global device
        device = torch.device('cuda')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = PIL.Image.fromarray(frame)
        frame = transforms.functional.to_tensor(frame).to(device)
        frame.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return frame[None, ...]
