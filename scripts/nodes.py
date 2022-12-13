import valens as va
from valens.nodes import *
from torch.multiprocessing import Process
import time

proxy = Process(target=va.bus.proxy)
proxy.start()

nodes = [
    VideoSource(max_fps=15, num_outputs=2),
    PoseFilter(),
    FeedbackFilter(output_frame=True, output_feedback=True, is_live=True),
    AwsSink()
]
for node in nodes:
    node.start()

for node in nodes:
    node.join()
