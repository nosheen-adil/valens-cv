from valens.stream import gen_addr_tcp

import zmq
import json
import numpy as np
import zmq.asyncio

def proxy(pub_port=6000, sub_port=6001):
    context = zmq.Context()

    frontend = context.socket(zmq.XPUB)
    frontend.bind(gen_addr_tcp(sub_port, bind=True))

    backend = context.socket(zmq.XSUB)
    backend.bind(gen_addr_tcp(pub_port, bind=True))

    zmq.proxy(frontend, backend)

    frontend.close()
    backend.close()
    context.term()

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


class MessageBus:
    def __init__(self, pub_port=6000, sub_port=6001):
        self.context = zmq.Context()
        self.pub_port = pub_port
        self.sub_port = sub_port
        self.pub_socket = None
        self.sub_socket = None
        self.sub_poller = None

        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.connect(gen_addr_tcp(pub_port, bind=False))
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(gen_addr_tcp(sub_port, bind=False))
        self.pub_socket.setsockopt(zmq.LINGER, 10)
        self.sub_socket.setsockopt(zmq.LINGER, 10)

        self.sub_poller = zmq.Poller()
        self.sub_poller.register(self.sub_socket, zmq.POLLIN)

        self.sub_messages = {}

    def subscribe(self, topic):
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, topic.encode('ascii'))
        self.sub_messages[topic] = []

    def _metadata(self, _type, _sync, **kwargs):
        metadata = {'type' : _type}
        if _sync is not None:
            metadata['sync'] = _sync
        if _type == 'ndarray':
            for key, value in kwargs.items():
                metadata[key] = value
        return metadata

    def send(self, topic, message=None, sync=None):
        self.pub_socket.send_string(topic, flags=zmq.SNDMORE)
        metadata = None
        if message is None:
            metadata = self._metadata('signal', sync)
            self.pub_socket.send_json(metadata)    
        elif type(message) is dict:
            metadata = self._metadata('json', sync)
            self.pub_socket.send_json(metadata, flags=zmq.SNDMORE)
            self.pub_socket.send_json(message)
        else:
            assert type(message) == np.ndarray
            metadata = self._metadata('ndarray', sync, dtype=str(message.dtype), shape=message.shape)
            self.pub_socket.send_json(metadata, flags=zmq.SNDMORE)
            self.pub_socket.send(message)

    def recv(self, topic, timeout=10):
        if len(self.sub_messages[topic]) > 0:
            return self.sub_messages[topic].pop(0)
        
        while True:
            socks = dict(self.sub_poller.poll(timeout=timeout))
            if self.sub_socket in socks and socks[self.sub_socket] == zmq.POLLIN:
                _topic = self.sub_socket.recv_string()
                metadata = self.sub_socket.recv_json()
                if metadata['type'] == 'signal':
                    message = True
                elif metadata['type'] == 'json':
                    message = self.sub_socket.recv_json()
                else:
                    assert metadata['type'] == 'ndarray'
                    raw = self.sub_socket.recv()
                    buf = memoryview(raw)
                    message = np.frombuffer(buf, dtype=metadata['dtype'])
                    message = np.reshape(message, metadata['shape'])

                if 'sync' in metadata:
                    self.sub_messages[_topic].append((metadata['sync'], message))
                else:
                    self.sub_messages[_topic].append(message)

                if topic == _topic:
                    break
            else:
                # if timeout and nothing from socket
                break
        
        if len(self.sub_messages[topic]) > 0:
            return self.sub_messages[topic].pop(0)
        return False


class AsyncMessageBus:
    def __init__(self, pub_port=6000, sub_port=6001):
        self.context = zmq.asyncio.Context()
        
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.connect(gen_addr_tcp(pub_port, bind=False))
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(gen_addr_tcp(sub_port, bind=False))
        self.pub_socket.setsockopt(zmq.LINGER, 10)
        self.sub_socket.setsockopt(zmq.LINGER, 10)

        self.sub_poller = zmq.asyncio.Poller()
        self.sub_poller.register(self.sub_socket, zmq.POLLIN)

        self.sub_messages = {}

    def subscribe(self, topic):
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, topic.encode('ascii'))
        self.sub_messages[topic] = []

    def _metadata(self, _type, _sync, **kwargs):
        metadata = {'type' : _type}
        if _sync is not None:
            metadata['sync'] = _sync
        if _type == 'ndarray':
            for key, value in kwargs.items():
                metadata[key] = value
        return metadata

    async def send(self, topic, message=None, sync=None):
        await self.pub_socket.send_string(topic, flags=zmq.SNDMORE)
        metadata = None
        if message is None:
            metadata = self._metadata('signal', None)
            await self.pub_socket.send_json(metadata)    
        elif type(message) is dict:
            metadata = self._metadata('json', sync)
            await self.pub_socket.send_json(metadata, flags=zmq.SNDMORE)
            await self.pub_socket.send_json(message)
        else:
            assert type(message) == np.ndarray
            metadata = self._metadata('ndarray', sync, dtype=str(message.dtype), shape=message.shape)
            await self.pub_socket.send_json(metadata, flags=zmq.SNDMORE)
            await self.pub_socket.send(message)

    async def recv(self, topic, timeout=10):
        if len(self.sub_messages[topic]) > 0:
            return self.sub_messages[topic].pop(0)
        
        while True:
            socks = dict(await self.sub_poller.poll(timeout=timeout))
            if self.sub_socket in socks and socks[self.sub_socket] == zmq.POLLIN:
                msg = await self.sub_socket.recv_multipart()
                _topic = msg[0].decode('utf-8')
                metadata = json.loads(msg[1].decode('utf-8'))
                if metadata['type'] == 'signal':
                    message = True
                elif metadata['type'] == 'json':
                    message = json.loads(msg[2].decode('utf-8'))
                else:
                    assert metadata['type'] == 'ndarray'
                    buf = memoryview(msg[2])
                    message = np.frombuffer(buf, dtype=metadata['dtype'])
                    message = np.reshape(message, metadata['shape'])

                if 'sync' in metadata:
                    self.sub_messages[_topic].append((metadata['sync'], message))
                else:
                    self.sub_messages[_topic].append(message)

                if topic == _topic:
                    break
            else:
                # if timeout and nothing from socket
                break
        
        if len(self.sub_messages[topic]) > 0:
            return self.sub_messages[topic].pop(0)
        return False

