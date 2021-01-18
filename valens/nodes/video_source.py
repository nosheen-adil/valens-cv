import cv2
from valens.structures.node import Node
import time
from valens.structures.stream import gen_addr_tcp, OutputStream

class VideoSource(Node):
    def __init__(self,
        output_address=gen_addr_tcp(),
        device_url=0):

        super().__init__("VideoSource")
        self.capture = cv2.VideoCapture(device_url)
        self.output_streams["frame"] = OutputStream(output_address)

    def process(self):
        start = time.time()
        ret, frame = self.capture.read()

        s = time.time()
        self.output_streams["frame"].send(frame)
        end = time.time()
        print("send time:", end - s)
