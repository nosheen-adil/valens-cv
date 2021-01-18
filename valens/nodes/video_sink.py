import cv2
from valens.structures.node import Node
import time
from valens.structures.stream import gen_addr_tcp, InputStream

class VideoSink(Node):
    def __init__(self,
        input_address=gen_addr_tcp()):
        super().__init__("VideoSink")
        self.input_streams["frame"] = InputStream(input_address)
    
    def process(self):
        start = time.time()
        frame = self.input_streams["frame"].recv()
        e = time.time()
        print("recv time:", e - start)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        end = time.time()
