import valens as va
from valens.stream import InputStream, OutputStream
import valens.bus
import valens.pipeline

from abc import ABC, abstractmethod
import time
from torch.multiprocessing import Process

class Node(ABC, Process):
    def __init__(self, name, topics=[], pub_port=6000, sub_port=6001, timeout=10):
        super().__init__()
        self.name = name
        self.input_streams = {}
        self.output_streams = {}
        self.max_fps = None
        self.total_time = 0
        self.iterations = 0
        self.stopped = False
        self.max_iterations = None

        self.bus = None
        self.pub_port = pub_port
        self.sub_port = sub_port
        self.topics = topics
        self.timeout = timeout

    def set_max_iterations(self, max_iterations):
        self.max_iterations = max_iterations

    def set_max_fps(self, max_fps):
        self.max_fps = max_fps
        
    def average_fps(self):
        if self.total_time == 0:
            return 0
        return self.iterations / self.total_time

    def run(self):
        self.load()
        print(self.name + ": running")
        # for _, stream in self.input_streams.items(): stream.start()
        # for _, stream in self.output_streams.items(): stream.start()

        self.bus = va.bus.MessageBus(self.pub_port, self.sub_port)
        time.sleep(0.5)
        self.bus.subscribe("signal")
        self.bus.subscribe("finished")
        for topic in self.topics:
            print(self.name + ': subscribing to ' + topic)
            self.bus.subscribe(topic)

        self.bus.send("active", {'name' : self.name})
        print(self.name, 'sent active signal')
        # self.prepare()

        if self.max_fps is not None:
            period = 1 / self.max_fps

        while not self.stopped:
            request = self.bus.recv("signal", timeout=None)
            signal = va.pipeline.Signal(request['signal'])
            if signal is va.pipeline.Signal.START:
                print(self.name + ': received start request', request)
                self.configure(request)
            else:
                print(self.name + ': stopping...')
                self.stopped = True

                for s in self.output_streams.values():
                    s.send()
                break

            while True:
                start = time.time()
                reset = self.process()
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

                finished = self.bus.recv("finished", timeout=None if reset is True else 10)
                if finished is True:
                    print(self.name + ': resetting...', reset is True)
                    self.reset()
                    self.total_time = 0
                    self.iterations = 0
                    self.stopped = False
                    # self.bus.reset()
                    # time.sleep(0.25)
                    # self.bus.subscribe("signal")
                    # self.bus.subscribe("finished")
                    # for topic in self.topics:
                    #     print(self.name + ': subscribing to ' + topic)
                    #     self.bus.subscribe(topic)
                    
                    break
                    # elif finished['level'] == 1:
                    #     print(self.name + ': stopping...')
                    #     self.stopped = True

                    #     for s in self.output_streams.values():
                    #         s.send()
                    #     break

        print(self.name + ": stopped with average fps of", self.average_fps())

    # def stop(self):
    #     self.stopped = True

    #     for s in self.output_streams.values():
    #         s.send()

    def load(self):
        pass

    @abstractmethod
    def process(self):
        pass

    def reset(self):
        pass

    def configure(self, request):
        pass

