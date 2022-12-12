from valens import stream

import pytest
import numpy as np
import time
from torch.multiprocessing import Process

def test_stream_gen_addr_inproc():
    address = stream.gen_addr_inproc("test")
    assert address == "inproc://test"

def test_stream_gen_addr_ipc():
    address = stream.gen_addr_ipc("test")
    assert address == "ipc:///tmp/test"

def test_stream_gen_addr_tcp_wildcard():
    address = stream.gen_addr_tcp()
    assert address == "tcp://*"

def test_stream_gen_addr_tcp():
    address = stream.gen_addr_tcp(5555)
    assert address == "tcp://localhost:5555"

def test_stream_gen_addr_tcp_bind():
    address = stream.gen_addr_tcp(5555, bind=True)
    assert address == "tcp://*:5555"

def test_stream_send_recv_single():
    address = stream.gen_addr_ipc("test")
    
    input_stream = stream.InputStream(address, identity=b'A')
    output_stream = stream.OutputStream(address, identities=[b'A'])

    def f_input(input_stream):
        input_stream.start()
        input_msg = {"hello" : "friend"}
        input_sync = {'user_id' : 'user', 'exercise' : 'exercise', 'set_id' : 'apples', 'timestamp' : 1000}
        output_msg, output_sync = input_stream.recv()
        assert input_msg == output_msg
        assert input_sync == output_sync
    
    def f_output(output_stream):
        output_stream.start()
        input_msg = {"hello" : "friend"}
        set_id = 'apples'
        timestamp = 1000
        sync = stream.gen_sync_metadata("user", "exercise", set_id, timestamp)
        output_stream.send(input_msg, sync)

    p = [Process(target=f_input, args=(input_stream,)),
        Process(target=f_output, args=(output_stream,))]
    
    [pi.start() for pi in p]
    [pi.join() for pi in p]

def test_stream_send_recv_multiple():
    address = stream.gen_addr_ipc("test")
    
    input_stream = stream.InputStream(address, identity=b'A')
    output_stream = stream.OutputStream(address, identities=[b'A'])

    def f_input(input_stream):
        input_stream.start()
        
        for i in range(100):
            msg, sync = input_stream.recv()
            assert msg["data"] == i
        
    def f_output(output_stream):
        output_stream.start()
        set_id = stream.gen_set_id()

        msg = {}
        for i in range(100):
            msg["data"] = i
            sync = stream.gen_sync_metadata("user", "exercise", set_id)
            output_stream.send(msg, sync)


    p = [Process(target=f_input, args=(input_stream,)),
        Process(target=f_output, args=(output_stream,))]
    
    [pi.start() for pi in p]
    [pi.join() for pi in p]

def test_stream_multiple_outputs():
    address = stream.gen_addr_ipc("test")

    input_stream1 = stream.InputStream(address, identity=b'A')
    input_stream2 = stream.InputStream(address, identity=b'B')
    output_stream = stream.OutputStream(address, identities=[b'A', b'B'])

    def f_input(input_stream):
        input_stream.start()
        
        for i in range(100):
            msg, sync = input_stream.recv()
            assert msg["data"] == i

    def f_output(output_stream):
        output_stream.start()
        time.sleep(0.25)
        set_id = stream.gen_set_id()
        sync = stream.gen_sync_metadata("user", "exercise", set_id)
        
        msg = {}
        for i in range(100):
            msg["data"] = i
            sync = stream.gen_sync_metadata("user", "exercise", set_id)
            output_stream.send(msg, sync)

    p = [Process(target=f_input, args=(input_stream1,)),
        Process(target=f_input, args=(input_stream2,)),
        Process(target=f_output, args=(output_stream,))]
    
    [pi.start() for pi in p]
    [pi.join() for pi in p]


def test_stream_send_recv_ndarray():
    address = stream.gen_addr_ipc("test")
    
    input_stream = stream.InputStream(address, identity='input')
    output_stream = stream.OutputStream(address, identities=['input'])

    def f_input(input_stream):
        input_stream.start()
        input_msg = np.array([[0, 1, 0], [1, 2, 3], [10, 9, 8]], dtype=np.int32)
        output_msg, sync = input_stream.recv()
        np.testing.assert_array_equal(input_msg, output_msg)

    def f_output(output_stream):
        output_stream.start()
        set_id = stream.gen_set_id()
        sync = stream.gen_sync_metadata("user", "exercise", set_id)
        
        input_msg = np.array([[0, 1, 0], [1, 2, 3], [10, 9, 8]], dtype=np.int32)
        output_stream.send(input_msg, sync)

    p = [Process(target=f_input, args=(input_stream,)),
        Process(target=f_output, args=(output_stream,))]
    
    [pi.start() for pi in p]
    [pi.join() for pi in p]
    