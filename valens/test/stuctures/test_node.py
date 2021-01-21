import pytest

from valens.structures.node import Node
from torch.multiprocessing import Event

import time

def test_stop_event():
    class TestNode(Node):
        def process(self):
            pass

    t = [TestNode("t1"), TestNode("t2")]

    stop_event = Event()
    for ti in t: ti.set_stop_event(stop_event)
    for ti in t: ti.start()
    time.sleep(10)
    stop_event.set()
    for ti in t: ti.join()

def test_terminate():
    class TestNode(Node):
        def process(self):
            pass

    t = [TestNode("t1"), TestNode("t2")]

    for ti in t: ti.start()
    time.sleep(10)
    for ti in t: ti.terminate()
