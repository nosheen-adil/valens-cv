import valens as va
import valens.pipeline
import valens.constants
from valens.nodes import *
from valens.stream import gen_addr_ipc

import pytest
from torch.multiprocessing import Process

def test_pipeline_start_stop_single():
    class Node(va.node.Node):
        def process(self):
            self.bus.send("finished")
            print('node: sent finished')

    def _pipeline():
        pipeline = va.pipeline.Pipeline()
        pipeline.add(Node('Node'))
        pipeline.loop()

    def _client():
        client = va.pipeline.Client()
        request = {
            'user_id' : 'test_0',
            'exercise' : 'BS',
            'num_reps' : -1,
            'capture' : va.constants.DATA_DIR + '/sequences/BS_test.h5'
        }
        client.start(request)
        client.wait_finish()
        client.stop()

    p = Process(target=_pipeline)
    c = Process(target=_client)

    p.start()
    c.start()

    p.join()
    c.join()

def test_pipeline_start_stop_multiple():
    class Node1(va.node.Node):
        def process(self):
            self.bus.send("finished")
            print('node: sent finished')

    class Node2(va.node.Node):
        def process(self):
            pass

    def _pipeline():
        pipeline = va.pipeline.Pipeline()
        pipeline.add(Node1('Node1'))
        pipeline.add(Node2('Node2'))
        pipeline.add(Node2('Node3'))
        pipeline.loop()

    def _client():
        client = va.pipeline.Client()
        request = {
            'user_id' : 'test_0',
            'exercise' : 'BS',
            'num_reps' : -1,
            'capture' : va.constants.DATA_DIR + '/sequences/BS_test.h5'
        }
        client.start(request)
        client.wait_finish()
        client.stop()

    p = Process(target=_pipeline)
    c = Process(target=_client)

    p.start()
    c.start()

    p.join()
    c.join()


def test_pipeline_pose_source_feedback_single():
    def _pipeline():
        pipeline = va.pipeline.Pipeline()
        pipeline.add(PoseSource())
        pipeline.add(FeedbackFilter())
        pipeline.add(Mp4FileSink(feedback_address=gen_addr_ipc('feedback'), output_dir=va.constants.TEST_DATA_DIR, fps=10))
        pipeline.loop()

    def _client():
        client = va.pipeline.Client()
        request = {
            'user_id' : 'test_0',
            'exercise' : 'BS',
            'num_reps' : -1,
            'capture' : va.constants.DATA_DIR + '/sequences/BS_test.h5',
            'name' : 'BS_test',
            'original_fps' : 30.0
        }
        client.start(request)
        client.wait_finish()
        client.stop()

    p = Process(target=_pipeline)
    c = Process(target=_client)

    p.start()
    c.start()

    p.join()
    c.join()
