from valens.structures.stream import InputStream, OutputStream
from abc import ABC, abstractmethod
import time
from multiprocessing import Process

class Node(ABC, Process):
    def __init__(self, name):
        super().__init__()

        self.name = name

        self.stop_event = None
        self.input_streams = {}
        self.output_streams = {}

    def configure_stop(self, stop_event):
        self.stop_event = stop_event

    def run(self):
        for _, stream in self.input_streams.items(): stream.start()
        for _, stream in self.output_streams.items(): stream.start()

        while self.stop_event is None or not self.stop_event.is_set():
            start = time.time()
            self.process()
            end = time.time()
            print(self.name + " fps:", 1 / (end - start))

    @abstractmethod
    def process(self):
        pass

