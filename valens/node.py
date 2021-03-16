from valens.stream import InputStream, OutputStream

from abc import ABC, abstractmethod
import time
from torch.multiprocessing import Process

class Node(ABC, Process):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.input_streams = {}
        self.output_streams = {}
        self.max_fps = None
        self.total_time = 0
        self.iterations = 0
        self.stopped = False
        self.max_iterations = None

    def stop(self):
        self.stopped = True

        for s in self.output_streams.values():
            s.send()

    def set_max_iterations(self, max_iterations):
        self.max_iterations = max_iterations

    def set_max_fps(self, max_fps):
        self.max_fps = max_fps
        
    def average_fps(self):
        if self.total_time == 0:
            return 0
        return self.iterations / self.total_time

    def run(self):
        print(self.name + ": running")
        for _, stream in self.input_streams.items(): stream.start()
        for _, stream in self.output_streams.items(): stream.start()
        self.prepare()
        if self.max_fps is not None:
            period = 1 / self.max_fps

        while not self.stopped:
            start = time.time()
            self.process()
            end = time.time()
            process_time = end - start
            if self.max_fps is not None:
                if process_time < period:
                    timeout = period - process_time
                    time.sleep(timeout)
                    process_time += timeout

            self.total_time += process_time
            self.iterations += 1
            # print(self.name + ": fps =", self.average_fps())

            if self.max_iterations and self.iterations > self.max_iterations:
                self.stop()
                break
        print(self.name + ": stopped with average fps of", self.average_fps())

    def prepare(self):
        pass

    @abstractmethod
    def process(self):
        pass
