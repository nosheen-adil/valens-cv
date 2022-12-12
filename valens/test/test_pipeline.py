import valens as va
import valens.pipeline
import valens.constants
from valens.nodes import *
from valens.stream import gen_addr_ipc

import pytest
from torch.multiprocessing import Process
import cv2
import random

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
        pipeline.add(VideoSource(max_fps=10))
        pipeline.add(PoseSource(max_fps=10))
        pipeline.add(FeedbackFilter())
        pipeline.add(Mp4FileSink(topics=['frame', 'feedback'], output_dir=va.constants.TEST_DATA_DIR, fps=10))
        pipeline.loop()

    def _client():
        client = va.pipeline.Client()
        request = {
            'user_id' : 'test_0',
            'set_id' : 'set_0',
            'exercise' : 'BS',
            'num_reps' : -1,
            'capture' : va.constants.DATA_DIR + '/recordings/BS_good_1.mp4',
            'name' : 'BS_good_1',
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

def test_pipeline_pose_source_feedback_multiple():
    def _pipeline():
        pipeline = va.pipeline.Pipeline()
        pipeline.add(VideoSource(max_fps=10))
        pipeline.add(PoseSource(max_fps=10))
        pipeline.add(FeedbackFilter())
        pipeline.add(Mp4FileSink(topics=['frame', 'feedback'], output_dir=va.constants.TEST_DATA_DIR, fps=10))
        pipeline.loop()

    def _client():
        cases = ['BS_good_1', 'BC_bad_1', 'PU_good_1']
        client = va.pipeline.Client()
        for i, case in enumerate(cases):
            request = {
                'user_id' : 'test_0',
                'set_id' : 'set_' + str(i),
                'exercise' : case[0:2],
                'num_reps' : 2,
                'capture' : va.constants.DATA_DIR + '/recordings/' + case + '.mp4',
                'name' : case,
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

def test_pipeline_video_source_feedback_single():
    def _pipeline():
        pipeline = va.pipeline.Pipeline()
        pipeline.add(VideoSource(max_fps=10))
        pipeline.add(PoseFilter())
        pipeline.add(FeedbackFilter())
        pipeline.add(Mp4FileSink(topics=['frame', 'feedback'], output_dir=va.constants.TEST_DATA_DIR, fps=10))
        pipeline.loop()

    def _client():
        client = va.pipeline.Client()
        request = {
            'user_id' : 'test_0',
            'set_id' : 'set_0',
            'exercise' : 'PU',
            'num_reps' : 2,
            'capture' : va.constants.DATA_DIR + '/recordings/PU_good_1.mp4',
            'name' : 'PU_good_1',
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

def test_pipeline_video_source_feedback_multiple():
    def _pipeline():
        pipeline = va.pipeline.Pipeline()
        pipeline.add(VideoSource(max_fps=10))
        pipeline.add(PoseFilter())
        pipeline.add(FeedbackFilter())
        pipeline.add(Mp4FileSink(topics=['frame', 'feedback'], output_dir=va.constants.TEST_DATA_DIR, fps=10))
        pipeline.loop()

    def _client():
        cases = ['BS_good_1', 'BC_bad_1', 'PU_good_1']
        client = va.pipeline.Client()
        for i, case in enumerate(cases):
            request = {
                'user_id' : 'test_0',
                'set_id' : 'set_' + str(i),
                'exercise' : case[0:2],
                'num_reps' : 2,
                'capture' : va.constants.DATA_DIR + '/recordings/' + case + '.mp4',
                'name' : case
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

def get_capture_fps(name):
    c = cv2.VideoCapture(name)
    return int(c.get(cv2.CAP_PROP_FPS))

def test_pipeline_aws_sink_feedback_bs():
    def _pipeline():
        pipeline = va.pipeline.Pipeline()
        pipeline.add(PoseSource(max_fps=10))
        pipeline.add(FeedbackFilter())
        pipeline.add(AwsSink(publish=False))
        pipeline.loop()

    def _client():
        cases = [
            'BS_good_1', 'BS_good_2', 'BS_good_3', 'BS_good_4',
            'BS_bad_1', 'BS_bad_2', 'BS_bad_3'
        ]
        random.shuffle(cases)
        client = va.pipeline.Client()
        for i, case in enumerate(cases):
            request = {
                'user_id' : 'demo_all',
                'set_id' : 'set_' + str(i),
                'exercise' : case[0:2],
                'num_reps' : -1,
                'capture' : va.constants.DATA_DIR + '/recordings/' + case + '.mp4',
                'name' : case,
                'original_fps' : get_capture_fps(va.constants.DATA_DIR + '/recordings/' + case + '.mp4')
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

def test_pipeline_aws_sink_feedback_bc():
    def _pipeline():
        pipeline = va.pipeline.Pipeline()
        pipeline.add(PoseSource(max_fps=10))
        pipeline.add(FeedbackFilter())
        pipeline.add(AwsSink(publish=False))
        pipeline.loop()

    def _client():
        cases = [
            'BC_good_1', 'BC_good_2', 'BC_good_3', 'BC_good_4', 'BC_good_5',
            'BC_bad_1', 'BC_bad_2', 'BC_bad_6'
        ]
        random.shuffle(cases)
        client = va.pipeline.Client()
        for i, case in enumerate(cases):
            request = {
                'user_id' : 'demo_all',
                'set_id' : 'set_' + str(i),
                'exercise' : case[0:2],
                'num_reps' : -1,
                'capture' : va.constants.DATA_DIR + '/recordings/' + case + '.mp4',
                'name' : case,
                'original_fps' : get_capture_fps(va.constants.DATA_DIR + '/recordings/' + case + '.mp4')
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

def test_pipeline_aws_sink_feedback_pu():
    def _pipeline():
        pipeline = va.pipeline.Pipeline()
        pipeline.add(PoseSource(max_fps=10))
        pipeline.add(FeedbackFilter())
        pipeline.add(AwsSink(publish=False))
        pipeline.loop()

    def _client():
        cases = [
            'PU_good_1', 'PU_good_2', 'PU_good_3',
            'PU_bad_1', 'PU_bad_4'
        ]
        random.shuffle(cases)
        client = va.pipeline.Client()
        for i, case in enumerate(cases):
            request = {
                'user_id' : 'demo_all',
                'set_id' : 'set_' + str(i),
                'exercise' : case[0:2],
                'num_reps' : -1,
                'capture' : va.constants.DATA_DIR + '/recordings/' + case + '.mp4',
                'name' : case,
                'original_fps' : get_capture_fps(va.constants.DATA_DIR + '/recordings/' + case + '.mp4')
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
