import valens
from valens.structures.stream import gen_addr_ipc
from valens.nodes import *
from valens import constants

import torch.multiprocessing
from torch.multiprocessing import set_start_method, Event

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    set_start_method('spawn')

    processes = [None] * 3
    frame_address = gen_addr_ipc("frame")
    keypoints_address = gen_addr_ipc("keypoints")
    video_path = constants.DATA_DIR + "/yoga_5min.mp4"
    model_path = constants.DATA_DIR + "/resnet18_baseline_att_224x224_A_epoch_249_trt.pth"
    pose_path = constants.DATA_DIR + "/human_pose.json"

    processes[0] = VideoSource(
                        frame_address=frame_address,
                        device_url=video_path,
                        num_outputs=2)
    processes[1] = PoseFilter(
                        frame_address=frame_address,
                        keypoints_address=keypoints_address,
                        model_path=model_path,
                        pose_path=pose_path)
    processes[2] = VideoSink(
                        frame_address=frame_address,
                        keypoints_address=keypoints_address,
                        pose_path=pose_path)
    processes[0].set_max_fps(10)

    stop_event = Event()
    for p in processes: p.set_stop_event(stop_event)

    import time
    processes[1].start() # it takes some time for the model to load
    time.sleep(40)
    processes[0].start()
    processes[2].start()

    time.sleep(180)
    print("Stopping processes")
    stop_event.set()
    for p in processes: p.join()
