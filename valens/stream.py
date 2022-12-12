from abc import ABC, abstractmethod
from base64 import b64encode
from enum import Enum
import numpy as np
import os
import time
import zmq
import zmq.asyncio
import json

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

def gen_addr_tcp(port=None, bind=False):
    if port is None:
        return gen_addr(Protocol.TCP, "*")
    
    assert (type(port) == int)
    if bind:
        return gen_addr(Protocol.TCP, "*:" + str(port))
    
    return gen_addr(Protocol.TCP, "localhost:" + str(port))

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
    def __init__(self, address, identity):
        super().__init__(address)
        self.identity = identity

    def start(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, self.identity)
        self.socket.connect(self.address)

    def recv(self):
        self.socket.send(b"req")
        x = self.socket.recv_multipart()
        md = json.loads(x[0].decode('utf-8'))
        sync = md["sync"]
        if (md["type"] == "ndarray"):
            buf = memoryview(x[1])
            A = np.frombuffer(buf, dtype=md['dtype']).reshape(md['shape'])
            return A, sync
        else:
            assert(md["type"] == "json")
            msg = json.loads(x[1].decode('utf-8'))
            return msg, sync

class OutputStream(Stream):
    def __init__(self, address, identities):
        super().__init__(address)
        assert type(identities) is list and len(identities) > 0
        self.identities = identities
        self.num_outputs = len(self.identities)
    
    def start(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(self.address)

    def send(self, msg=None, sync=None):
        remaining = self.identities.copy()
        while len(remaining) > 0:
            identity, req = self.socket.recv_multipart()
            assert req == b"req" and identity in remaining
            remaining.remove(identity)

        for i, identity in enumerate(self.identities):
            if type(msg) == np.ndarray:
                md = dict(
                    type = "ndarray",
                    dtype = str(msg.dtype),
                    shape = msg.shape,
                    sync = sync
                )
                self.socket.send_multipart([identity, json.dumps(md).encode('utf-8'), msg])
            else:
                md = dict(
                    type = "json",
                    sync = sync
                )
                self.socket.send_multipart([identity, json.dumps(md).encode('utf-8'), json.dumps(msg).encode('utf-8')])

class AsyncInputStream(Stream):
    def __init__(self, address, identity):
        super().__init__(address)
        self.identity = identity

    def start(self):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, self.identity)
        self.socket.connect(self.address)

    async def recv(self):
        await self.socket.send(b"req")
        x = await self.socket.recv_multipart()
        md = json.loads(x[0].decode('utf-8'))
        sync = md["sync"]
        if (md["type"] == "ndarray"):
            buf = memoryview(x[1])
            A = np.frombuffer(buf, dtype=md['dtype']).reshape(md['shape'])
            return A, sync
        else:
            assert(md["type"] == "json")
            msg = json.loads(x[1].decode('utf-8'))
            return msg, sync

class AsyncOutputStream(Stream):
    def __init__(self, address, identities):
        super().__init__(address)
        assert type(identities) is list and len(identities) > 0
        self.identities = identities
        self.num_outputs = len(self.identities)
    
    def start(self):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(self.address)

    async def send(self, msg=None, sync=None):
        remaining = self.identities.copy()
        while len(remaining) > 0:
            identity, req = await self.socket.recv_multipart()
            assert req == b"req" and identity in remaining
            remaining.remove(identity)

        for i, identity in enumerate(self.identities):
            if type(msg) == np.ndarray:
                md = dict(
                    type = "ndarray",
                    dtype = str(msg.dtype),
                    shape = msg.shape,
                    sync = sync
                )
                await self.socket.send_multipart([identity, json.dumps(md).encode('utf-8'), msg])
            else:
                md = dict(
                    type = "json",
                    sync = sync
                )
                await self.socket.send_multipart([identity, json.dumps(md).encode('utf-8'), json.dumps(msg).encode('utf-8')])
