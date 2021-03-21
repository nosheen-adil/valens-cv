from valens.stream import gen_addr_tcp

import zmq
import json

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

def _encode_topic_message(topic, message=None):
    if message is None:
        return topic + ';'
    return ";".join([topic, json.dumps(message)])

def _decode_topic_message(raw):
    index = raw.find(';')
    topic = raw[0:index]
    if index == (len(raw) - 1):
        return topic, None
    return topic, json.loads(raw[index+1:])


class MessageBus:
    def __init__(self, pub_port=6000, sub_port=6001):
        self.context = zmq.Context()
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.connect(gen_addr_tcp(pub_port, bind=False))
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(gen_addr_tcp(sub_port, bind=False))

        self.sub_poller = zmq.Poller()
        self.sub_poller.register(self.sub_socket, zmq.POLLIN)

        self.sub_messages = {}

    def subscribe(self, topic):
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, topic.encode('ascii'))
        self.sub_messages[topic] = []

    def send(self, topic, message=None):
        self.pub_socket.send_string(_encode_topic_message(topic, message))

    def _load_dict(self, timeout=10):
        while True:
            # process all data which is immediately available at socket
            socks = dict(self.sub_poller.poll(timeout=timeout))
            if self.sub_socket in socks and socks[self.sub_socket] == zmq.POLLIN:
                raw = self.sub_socket.recv_string()
                topic, message = _decode_topic_message(raw)
                self.sub_messages[topic].append(message)
                # print(topic, message)

                if timeout is None or timeout == 0:
                    return
            else:
                return

    def recv(self, topic, timeout=10):
        if timeout is None:
            if len(self.sub_messages[topic]) > 0:
                _message = self.sub_messages[topic].pop(0)
                if _message is None:
                    return True
                else:
                    return _message
        
        self._load_dict(timeout)

        if len(self.sub_messages[topic]) > 0:
            _message = self.sub_messages[topic].pop(0)
            if _message is None:
                return True
            else:
                return _message
        return False
