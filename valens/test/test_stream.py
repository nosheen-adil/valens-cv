from valens import stream

import pytest
import numpy as np
import time
from torch.multiprocessing import Process

def test_gen_addr_inproc():
    address = stream.gen_addr_inproc("test")
    assert address == "inproc://test"

def test_gen_addr_ipc():
    address = stream.gen_addr_ipc("test")
    assert address == "ipc:///tmp/test"

def test_gen_addr_tcp_wildcard():
    address = stream.gen_addr_tcp()
    assert address == "tcp://*"

def test_gen_addr_tcp():
    address = stream.gen_addr_tcp(5555)
    assert address == "tcp://5555"

def test_send_recv():
    address = stream.gen_addr_ipc("test")
    
    input_stream = stream.InputStream(address)
    output_stream = stream.OutputStream(address)

    def f_input(input_stream):
        input_stream.start()
        input_msg = {"hello" : "friend"}
        output_msg = input_stream.recv()
        assert input_msg == output_msg
    
    def f_output(output_stream):
        output_stream.start()
        input_msg = {"hello" : "friend"}
        output_stream.send(input_msg)

    p = [Process(target=f_input, args=(input_stream,)),
        Process(target=f_output, args=(output_stream,))]
    
    [pi.start() for pi in p]
    [pi.join() for pi in p]

def test_send_recv_multiple():
    address = stream.gen_addr_ipc("test")
    
    input_stream = stream.InputStream(address)
    output_stream = stream.OutputStream(address)

    def f_input(input_stream):
        input_stream.start()
        
        msg = input_stream.recv()
        assert msg["data"] == 1

        msg = input_stream.recv()
        assert msg["data"] == 2

        msg = input_stream.recv()
        assert msg["data"] == 3
        
    def f_output(output_stream):
        output_stream.start()
        
        msg = {"data" : 1}
        output_stream.send(msg)

        msg["data"] = 2
        output_stream.send(msg)

        msg["data"] = 3
        output_stream.send(msg)

    p = [Process(target=f_input, args=(input_stream,)),
        Process(target=f_output, args=(output_stream,))]
    
    [pi.start() for pi in p]
    [pi.join() for pi in p]

def test_multiple_outputs():
    address = stream.gen_addr_ipc("test")

    input_stream1 = stream.InputStream(address)
    input_stream2 = stream.InputStream(address)
    output_stream = stream.OutputStream(address, num_outputs=2)

    def f_input(input_stream):
        input_stream.start()
        
        msg = input_stream.recv()
        assert msg["data"] == 1

    def f_output(output_stream):
        output_stream.start()
        
        msg = {"data" : 1}
        output_stream.send(msg)

    p = [Process(target=f_input, args=(input_stream1,)),
        Process(target=f_input, args=(input_stream2,)),
        Process(target=f_output, args=(output_stream,))]
    
    [pi.start() for pi in p]
    [pi.join() for pi in p]


def test_send_recv_ndarray():
    address = stream.gen_addr_ipc("test")
    
    input_stream = stream.InputStream(address)
    output_stream = stream.OutputStream(address)

    def f_input(input_stream):
        input_stream.start()
        input_msg = np.array([[0, 1, 0], [1, 2, 3], [10, 9, 8]], dtype=np.int32)
        output_msg = input_stream.recv()
        np.testing.assert_array_equal(input_msg, output_msg)

    def f_output(output_stream):
        output_stream.start()
        input_msg = np.array([[0, 1, 0], [1, 2, 3], [10, 9, 8]], dtype=np.int32)
        output_stream.send(input_msg)

    p = [Process(target=f_input, args=(input_stream,)),
        Process(target=f_output, args=(output_stream,))]
    
    [pi.start() for pi in p]
    [pi.join() for pi in p]
    