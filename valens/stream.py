from abc import ABC, abstractmethod
from base64 import b64encode
from enum import Enum
import numpy as np
import os
import time
import zmq

class Protocol(Enum):
    INPROC="inproc"
    IPC="ipc"
    TCP="tcp"

def gen_addr(transport, endpoint):
    assert(type(transport) == Protocol)
    return transport.value + "://" + endpoint

def gen_addr_inproc(endpoint):
    return gen_addr(Protocol.INPROC, endpoint)

def gen_addr_ipc(endpoint):
    return gen_addr(Protocol.IPC, "/tmp/" + endpoint)

def gen_addr_tcp(port=None):
    if port is None:
        return gen_addr(Protocol.TCP, "*")
    
    assert (type(port) == int)
    return gen_addr(Protocol.TCP, str(port))

def gen_set_id(size=5):
    return b64encode(os.urandom(size)).decode('utf-8')

def gen_sync_metadata(user_id, exercise, set_id, timestamp=None):
    if timestamp is None:
        timestamp = int(time.time() * 1000)
    return {
        "user_id" : user_id,
        "exercise" : exercise,
        "timestamp" : timestamp,
        "set_id" : set_id
    }

class Stream(ABC):
    def __init__(self, address):
        self.address = address
        self.context = None
        self.socket = None

    @abstractmethod
    def start(self):
        pass

class InputStream(Stream):
    def __init__(self, address):
        super().__init__(address)

    def start(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.address)

    def recv(self):
        self.socket.send(b"req")

        md = self.socket.recv_json()
        sync = md["sync"]
        if (md["type"] == "ndarray"):
            return self.recv_array(md), sync
        else:
            assert(md["type"] == "json")
            msg = self.socket.recv_json()
            return msg, sync
        
    def recv_array(self, md, flags=0, copy=True, track=False):
        """recv a numpy array"""
        msg = self.socket.recv(flags=flags, copy=copy, track=track)
        buf = memoryview(msg)
        A = np.frombuffer(buf, dtype=md['dtype'])
        return A.reshape(md['shape'])

class OutputStream(Stream):
    def __init__(self, address, num_outputs=1):
        super().__init__(address)

        self.num_outputs = num_outputs
    
    def start(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.address)

    def send(self, msg=None, sync=None):
        for i in range(self.num_outputs):
            req = self.socket.recv()
            if type(msg) == np.ndarray:
                self.send_array(msg, sync, copy=False)
            else:
                self.send_json(msg, sync)

    def send_array(self, A, sync=None, flags=0, copy=True, track=False):
        """send a numpy array with metadata"""
        md = dict(
            type = "ndarray",
            dtype = str(A.dtype),
            shape = A.shape,
            sync = sync
        )
        self.socket.send_json(md, flags|zmq.SNDMORE)
        return self.socket.send(A, flags, copy=copy, track=track)

    def send_json(self, j, sync=None, flags=0):
        md = dict(
            type = "json",
            sync = sync
        )

        self.socket.send_json(md, flags|zmq.SNDMORE)
        self.socket.send_json(j)
