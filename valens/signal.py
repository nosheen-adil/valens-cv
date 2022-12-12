import zmq
import zmq.asyncio
from enum import Enum
from valens.stream import gen_addr_ipc

class Signal(Enum):
    start = 'start'
    reset = 'reset'
    stop = 'stop'

class SignalSender:
    def __init__(self, address=gen_addr_ipc('signal')):
        self.address = address

    def start(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(self.address)

    def send(self, signal):
        self.socket.send_string(signal.value)

class SignalReceiver:
    def __init__(self, address=gen_addr_ipc('signal')):
        self.address = address

    def start(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.connect(self.address)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
    
    def recv(self, timeout=None):
        socks = dict(self.poller.poll(timeout=timeout))
        if self.socket in socks and socks[self.socket] == zmq.POLLIN:
            msg = self.socket.recv_string()
            return Signal(msg)

class AsyncSignalSender:
    def __init__(self, address=gen_addr_ipc('signal')):
        self.address = address

    def start(self):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(self.address)

    async def send(self, signal):
        await self.socket.send_string(signal.value)

class AsyncSignalReceiver:
    def __init__(self, address=gen_addr_ipc('signal')):
        self.address = address

    def start(self):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.connect(self.address)
        self.poller = zmq.asyncio.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
    
    async def recv(self, timeout=None):
        socks = dict(await self.poller.poll(timeout=timeout))
        if self.socket in socks and socks[self.socket] == zmq.POLLIN:
            msg = await self.socket.recv_string()
            return Signal(msg)
