from valens import bus

import pytest
from torch.multiprocessing import Process
import time
import numpy as np

def test_bus_send_recv_signal():
    proxy = Process(target=bus.proxy)
    proxy.start()

    bus1 = bus.MessageBus()
    bus2 = bus.MessageBus()

    bus1.subscribe("start")
    time.sleep(0.25)
    
    bus2.send("start")

    message = bus1.recv("start", timeout=None)
    assert message == True

    proxy.terminate()

def test_bus_send_recv_json():
    proxy = Process(target=bus.proxy)
    proxy.start()

    bus1 = bus.MessageBus()
    bus2 = bus.MessageBus()

    bus1.subscribe("start")
    time.sleep(0.25)
    
    bus2.send("start", {'user_id' : 'test_0', 'exercise' : 'push-up', 'num_reps' : 5})

    message = bus1.recv("start", timeout=None)
    assert message == {'user_id' : 'test_0', 'exercise' : 'push-up', 'num_reps' : 5}

    proxy.terminate()

def test_bus_send_recv_ndarray():
    proxy = Process(target=bus.proxy)
    proxy.start()

    bus1 = bus.MessageBus()
    bus2 = bus.MessageBus()

    bus1.subscribe("start")
    time.sleep(0.25)
    
    bus2.send("start", np.array([[0, 1, 0], [1, 2, 3], [10, 9, 8]], dtype=np.int32))

    message = bus1.recv("start", timeout=None)
    np.testing.assert_equal(message, np.array([[0, 1, 0], [1, 2, 3], [10, 9, 8]], dtype=np.int32))

    proxy.terminate()

def test_bus_send_recv_sync():
    proxy = Process(target=bus.proxy)
    proxy.start()

    bus1 = bus.MessageBus()
    bus2 = bus.MessageBus()

    bus1.subscribe("start")
    time.sleep(0.25)
    
    sync = bus.gen_sync_metadata('test_0', 'push-up', 'set_0', 1000)
    bus2.send("start", {'user_id' : 'test_0', 'exercise' : 'push-up', 'num_reps' : 5}, sync)

    sync, message = bus1.recv("start", timeout=None)
    assert message == {'user_id' : 'test_0', 'exercise' : 'push-up', 'num_reps' : 5}
    assert sync == {'user_id': 'test_0', 'exercise': 'push-up', 'timestamp': 1000, 'set_id': 'set_0'}

    proxy.terminate()

def test_bus_recv_no_message():
    proxy = Process(target=bus.proxy)
    proxy.start()

    bus1 = bus.MessageBus()
    bus2 = bus.MessageBus()

    bus1.subscribe("start")
    time.sleep(0.25)
    
    bus2.send("stop")

    message = bus1.recv("start")
    assert message == False

    proxy.terminate()

def test_bus_recv_topic_later():
    proxy = Process(target=bus.proxy)
    proxy.start()

    bus1 = bus.MessageBus()
    bus2 = bus.MessageBus()

    bus1.subscribe("start")
    bus1.subscribe("active")
    time.sleep(0.25)
    
    bus2.send("start")

    message = bus1.recv("active")
    assert message == False

    bus2.send("active", {'name' : 'Node'})

    message = bus1.recv("start")
    assert message == True

    message = bus1.recv("active")
    assert message == {'name' : 'Node'}

    proxy.terminate()
    