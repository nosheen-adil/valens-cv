import valens as va
from valens.pipeline import Pipeline
from valens.nodes import *
import asyncio
from torch.multiprocessing import Process
import time

pipeline = Pipeline()
# pipeline.names = ['VideoSource', 'PoseFilter', 'FeedbackFilter']
pipeline.run()
