import valens as va
from valens import constants
from valens.node import Node
import valens.pose
from valens.stream import InputStream, OutputStream, gen_addr_ipc

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
    def __init__(self, model_path=constants.POSE_MODEL_TRT_WEIGHTS, pose_path=constants.POSE_JSON):
        super().__init__("PoseFilter")
        self.model_path = model_path
        self.pose_path = pose_path
        self.model_trt = None
        self.mean = None
        self.std = None
        self.prev = None
        self.set_id = None

        self.input_streams['frame'] = InputStream(gen_addr_ipc('frame'), identity=b'0')
        self.output_streams['pose'] = OutputStream(gen_addr_ipc('pose'), identities=[b'0'])

    def reset(self):
        self.prev = None
        self.set_id = None

    def configure(self, request):
        self.set_id = request['set_id']

    def load(self):
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

        data = torch.zeros((1, 3, constants.POSE_MODEL_WIDTH, constants.POSE_MODEL_HEIGHT)).cuda()
        for _ in range(5):
            self.model_trt(data)

    def process(self):
        frame, sync = self.input_streams['frame'].recv()
        if frame is None:
            print('PoseFilter: detected frame sentinel')
            self.output_streams['pose'].send()
            return True
        
        # print('PoseFilter: sending')
        frame = cv2.resize(frame, dsize=(constants.POSE_MODEL_WIDTH, constants.POSE_MODEL_HEIGHT))
        data = self.preprocess(frame)
        cmap, paf = self.model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)
        # print('PoseFilter: stopping')
        pose = va.pose.nn_to_numpy(counts, objects, peaks, prev=self.prev)
        self.prev = pose.copy()
        sync['fps'] = self.average_fps()
        self.output_streams['pose'].send(pose, sync)

    def preprocess(self, frame):
        global device
        device = torch.device('cuda')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = PIL.Image.fromarray(frame)
        frame = transforms.functional.to_tensor(frame).to(device)
        frame.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return frame[None, ...]
