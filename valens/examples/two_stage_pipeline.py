import valens
from valens.structures.stream import InputStream, OutputStream, gen_addr_ipc, gen_addr_tcp
from valens.nodes import *
from multiprocessing import Process, Event, Pipe
from threading import Thread

stop_event = Event()
processes = [None] * 2
address = gen_addr_ipc("test")

processes[0] = VideoSource(output_address=address)
processes[1] = VideoSink(input_address=address)

for process in processes:
    process.start()

for process in processes:
    process.join()
