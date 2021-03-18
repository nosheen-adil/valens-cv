from valens.node import Node
from valens.stream import gen_set_id, gen_sync_metadata, gen_addr_ipc, InputStream, OutputStream

import pytest
import time

def test_node_sentinel_stop():
    class Source(Node):
        def __init__(self, name, total_count=5, output_addr=gen_addr_ipc("test")):
            super().__init__(name)
            self.total_count = total_count
            self.count = 0
            self.output_streams["out"] = OutputStream(output_addr)

        def process(self):
            if self.count == self.total_count:
                self.stop()
                return
            
            set_id = gen_set_id()
            sync = gen_sync_metadata("user", "exercise", set_id)
            self.output_streams["out"].send({"hey":"there"}, sync)
            self.count += 1
            time.sleep(0.1)

    class Sink(Node):
        def __init__(self, name, input_addr=gen_addr_ipc("test")):
            super().__init__(name)
            self.input_streams["in"] = InputStream(input_addr)

        def process(self):
            result, sync = self.input_streams["in"].recv()
            if result is None:
                self.stop()
                return

    t = [Source("source", total_count=10), Sink("sink")]
    for ti in t: ti.start()
    for ti in t: ti.join()
