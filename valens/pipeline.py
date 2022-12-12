import valens as va
import valens.bus
import valens.stream

from aiohttp import web, MultipartWriter
from PIL import Image
import base64
import io
import numpy as np
import cv2
import time
import json
import aiohttp_cors
import asyncio

import json
import numpy as np
import random
import imutils

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

class Pipeline:
    def __init__(self, http_port=8080, pub_port=6000, sub_port=6001):
        self.bus = va.bus.AsyncMessageBus(pub_port=pub_port, sub_port=sub_port)
        time.sleep(0.5)
        # self.bus.subscribe("active")
        self.bus.subscribe("finished")
        self.bus.subscribe('feedback_frame')

        # self.frame_stream = va.stream.AsyncInputStream(va.stream.gen_addr_tcp(7000, bind=False), identity=b'0')
        # self.frame_stream.start()

        self.nodes = []
        self.names = []

        self.width = 640
        self.height = 480
        self.frame = None
        # np.zeros((self.height, self.width, 3), np.uint8)

        self.capturing = False
        self.finished = False
        self.http_port = http_port

        # self.cases = {
        #     'squats' : ('BS', ['BS_good_1', 'BS_good_2', 'BS_good_3', 'BS_good_4', 'BS_bad_1', 'BS_bad_2', 'BS_bad_3']),
        #     'bicepcurls': ('BC', ['BC_good_1', 'BC_good_2', 'BC_good_3', 'BC_good_4', 'BC_good_5', 'BC_bad_1', 'BC_bad_2']),
        #     'pushups' : ('PU', ['PU_good_1', 'PU_good_2', 'PU_good_3', 'PU_bad_1', 'PU_bad_4'])
        # }

        self.cases = {
            'squats' : {
                'code': 'BS',
                'names' : ['BS_bad_4', 'BS_bad_1', 'BS_good_1', 'BS_good_2'],
                'count' : 0
            },
            'bicepcurls': {
                'code': 'BC',
                'names' : ['BC_bad_1', 'BC_bad_2', 'BC_good_1', 'BC_good_2'],
                'count' : 0
            },
            'pushups' : {
                'code': 'PU',
                'names' : ['PU_good_1', 'PU_bad_1'],
                'count' : 0
            },
        }

        self.set_id = None

    def add(self, node):
        self.nodes.append(node)
        self.names.append(node.name)

    def run(self):
        app = web.Application()
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                )
        })

        start_resource = cors.add(app.router.add_resource("/start"))
        cors.add(start_resource.add_route("POST", self.start_handle))

        # capture_resource = cors.add(app.router.add_resource("/capture"))
        # cors.add(capture_resource.add_route("POST", self.capture_handle))

        finished_resource = cors.add(app.router.add_resource("/finished"))
        cors.add(finished_resource.add_route("GET", self.finished_handle))

        feedback_resource = cors.add(app.router.add_resource("/feedback.mjpeg"))
        cors.add(feedback_resource.add_route("GET", self.feedback_handle))

        web.run_app(app, port=self.http_port)

    # async def load(self):
    #     # for node in self.nodes:
    #     #     node.start()

    #     print('Pipeline: waiting for nodes to activate')
    #     names = self.names.copy()
    #     while len(names):
    #         active = await self.bus.recv('active', timeout=None)
    #         assert 'name' in active
    #         names.remove(active['name'])
    #         print('names', names)
    #     print("Pipeline: all nodes active")        
 
    async def start_handle(self, request):
        def get_capture_fps(name):
            c = cv2.VideoCapture(name)
            return int(c.get(cv2.CAP_PROP_FPS))

        # if self.activated is False:
        #     await self.load()
        #     self.activated = True

        if self.capturing is False:
            result = web.json_response({'success' : True})
            data = await request.json()
            print('start request', data)

            label = data['exercise']
            assert label in self.cases
            case = self.cases[label]['names'][self.cases[label]['count']]
            print('case: ', self.cases[label]['count'])
            self.cases[label]['count'] = (self.cases[label]['count'] + 1) % len(self.cases[label]['names'])
            data['exercise'] = self.cases[label]['code']
            data['signal'] = 'start'
            data['capture'] = va.constants.DATA_DIR + '/recordings/shashank/' + case + '.mp4'
            data['name'] = case
            data['original_fps'] = get_capture_fps(data['capture'])
            data['publish'] = True
            # data['user_id'] = 'test_1'

            self.capturing = True
            self.finished = False
            self.set_id = data['set_id']

            await self.bus.send("signal", data)
            print('Pipeline: sent signal')
        else:
            result = web.json_response({'success' : False})
        return result

    # async def capture_handle(self, request):
    #     if self.capturing:
    #         start = time.time()
    #         result = web.json_response({'success' : True})
    #         data = await request.json()
    #         post_data = data['image']
    #         base64_encoded = post_data[22:]
    #         base64_decoded = base64.b64decode(base64_encoded)
    #         image = Image.open(io.BytesIO(base64_decoded))
    #         image = np.array(image)
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #         sync = {
    #             'user_id' : 'test_0',
    #             'set_id' : 'set_0',
    #             'exercise' : 'PU'
    #         }
    #         # print('sending frame')
    #         await self.bus.send('frame', image, sync)
    #         # print('sent frame')
    #         # self.frame = image
    #         end = time.time()
    #         print('fps:', 1/(end-start))
    #     else:
    #         print('got capture request before start')
    #         result = web.json_response({'success' : False})
    #     return result

    async def finished_handle(self, request):
        if not self.finished:
            finished = await self.bus.recv('finished')
            if finished:
                self.finished = True

        if self.capturing:
            frame = await self.bus.recv('feedback_frame', timeout=None if not self.finished else 30)
            if frame is False:
                self.capturing = False
                self.frame = None
                self.finished = True
            else:
                self.frame = frame
        finished = self.finished and not self.capturing
        return web.json_response({'finished' : finished})

    async def feedback_handle(self, request):
        my_boundary = 'image-boundary'
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'multipart/x-mixed-replace;boundary={}'.format(my_boundary)
            }
        )
        await response.prepare(request)
        while True:
            await asyncio.sleep(0.1)
            # frame = await self.bus.recv('feedback_frame', timeout=30)
            # if frame is False:
            #     frame = np.zeros((self.height, self.width, 3), np.uint8)
            #     if self.capturing:
            #         self.capturing = False
            frame = self.frame
            if frame is None:
                frame = np.zeros((self.height, self.width, 3), np.uint8)
            
            _, frame = cv2.imencode('.jpg', frame)
            frame = frame.tostring()
            with MultipartWriter('image/jpeg', boundary=my_boundary) as mpwriter:
                mpwriter.append(frame, {
                    'Content-Type': 'image/jpeg'
                })
                try:
                    await mpwriter.write(response, close_boundary=False)
                except ConnectionResetError :
                    break
            await response.drain()
