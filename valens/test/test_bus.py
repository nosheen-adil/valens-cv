from valens import bus

import pytest
from torch.multiprocessing import Process
import time

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

def test_bus_send_recv_message():
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
    