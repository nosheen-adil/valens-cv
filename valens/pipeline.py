import valens as va
from valens.stream import gen_addr_tcp
import valens.bus

import zmq
from enum import Enum
from torch.multiprocessing import Process
import time

class State(Enum):
    IDLE = 'idle'
    RUNNING = 'running'

class Signal(Enum):
    START = 'start'
    STOP = 'stop'

def _check_request(request, keys):
    if 'signal' not in request:
        return {'success' : False, 'error' : "expected signal"}
    elif Signal(request['signal']) is Signal.STOP:
        return {'success' : True}
    
    if set(keys).issubset(set(request.keys())):
        return {'success' : True}
    else:
        return {'success' : False, 'error' : "expected keys: " + str(k)}

class Client:
    def __init__(self, start_port=5500, stop_port=5501):
        self.context = zmq.Context()
        self.signal_socket = self.context.socket(zmq.REQ)
        self.signal_socket.connect(gen_addr_tcp(start_port, bind=False))
        self.finish_socket = self.context.socket(zmq.REP)
        self.finish_socket.bind(gen_addr_tcp(stop_port, bind=True))

    def start(self, request):
        request['signal'] = Signal.START.value
        self.request = request
        self.signal_socket.send_json(request)
        response = self.signal_socket.recv_json()
        if response['success'] is True:
            print('Client: successfully started pipeline')
        else:
            print('Client: error with start request')

    def wait_finish(self):
        print('Cleint: waiting for request')
        request = self.finish_socket.recv_json()
        print('Client: received request', request)
        response = _check_request(request, keys=self.request.keys())
        self.finish_socket.send_json(response)
        if response['success'] is True:
            print('Client: detected pipeline is finished')
        else:
            print('Client: error with stop request')
        self.request = None

    def stop(self):
        request = {
            'signal' : Signal.STOP.value
        }
        self.signal_socket.send_json(request)
        response = self.signal_socket.recv_json()
        if response['success'] is True:
            print('Client: successfully stopped pipeline')
        else:
            print('Client: failed to stop pipeline')
        

class Pipeline:
    def __init__(self, start_port=5500, stop_port=5501, pub_port=6000, sub_port=6001):
        self.context = zmq.Context()
        self.signal_socket = self.context.socket(zmq.REP)
        self.signal_socket.bind(gen_addr_tcp(start_port, bind=True))
        self.finish_socket = self.context.socket(zmq.REQ)
        self.finish_socket.connect(gen_addr_tcp(stop_port, bind=False))

        self.proxy = Process(target=va.bus.proxy, args=(pub_port, sub_port))
        self.proxy.start()

        self.bus = va.bus.MessageBus(pub_port=pub_port, sub_port=sub_port)
        time.sleep(0.5)
        self.bus.subscribe("active")
        self.bus.subscribe("finished")

        self.nodes = []
        self.names = []

    def add(self, node):
        self.nodes.append(node)
        self.names.append(node.name)

    def loop(self):
        for node in self.nodes:
            node.start()

        print('Pipeline: waiting for nodes to activate')
        names = self.names.copy()
        while len(names):
            active = self.bus.recv("active", timeout=None)
            assert 'name' in active
            names.remove(active['name'])
            print('names', names)
        print("Pipeline: all nodes active")

        while True:
            request = self.signal_socket.recv_json()
            print('Received request:', request)
            response = _check_request(request, keys=['signal', 'user_id', 'exercise', 'num_reps'])
            self.signal_socket.send_json(response)
            if response['success'] is False:
                print('Pipeline: bad start request')
                continue

            signal = Signal(request['signal'])
            self.bus.send("signal", request)
            print('Pipeline: sent signal')
                
            if signal is Signal.STOP:
                print('Pipeline: stopping loop')
                break
            
            finished = self.bus.recv("finished", timeout=None)
            assert finished is True

            print('Finished set!', finished)
            request['finished'] = True
            print('Pipeline: sending stop request')
            self.finish_socket.send_json(request)
            print('Pipeline: recevied stop response')
            
            response = self.finish_socket.recv_json()
            if response['success'] is False:
                print('Pipeline: bad stop response')
        
        for node in self.nodes:
            node.join()
        self.proxy.terminate()
