from valens.structures.node import Node
from valens.structures.stream import InputStream, OutputStream
from valens.structures import pose

import torch
import torchvision.transforms as transforms
import trt_pose.coco
from trt_pose.parse_objects import ParseObjects
import json
from torch2trt import TRTModule
import cv2
import PIL.Image
import numpy as np

class PoseFilter(Node):
    def __init__(self, frame_address, keypoints_address, model_path, pose_path):
        super().__init__("PoseFilter")
        self.input_streams["frame"] = InputStream(frame_address)
        self.output_streams["pose"] = OutputStream(keypoints_address)

        self.model_path = model_path
        self.pose_path = pose_path

    def prepare(self):
        self.parse_objects = ParseObjects(pose.get_topology(self.pose_path))

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
        k = pose.trt_to_json(counts, objects, peaks)
        self.output_streams["pose"].send(k)
        
    def preprocess(self, frame):
        global device
        device = torch.device('cuda')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = PIL.Image.fromarray(frame)
        frame = transforms.functional.to_tensor(frame).to(device)
        frame.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return frame[None, ...]
