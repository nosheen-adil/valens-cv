from valens import constants
from valens import exercise
from valens import pose
from valens import feedback
from valens.node import Node
from valens.stream import InputStream, gen_addr_ipc

import valens as va

from abc import abstractmethod
import cv2
import json
import trt_pose.coco
import numpy as np

from awscrt import io, mqtt, auth, http
from awsiot import mqtt_connection_builder
import sys
import threading
from uuid import uuid4
import time

# Callback when connection is accidentally lost.
def on_connection_interrupted(connection, error, **kwargs):
    print("Connection interrupted. error: {}".format(error))


# Callback when an interrupted connection is re-established.
def on_connection_resumed(connection, return_code, session_present, **kwargs):
    print("Connection resumed. return_code: {} session_present: {}".format(return_code, session_present))

    if return_code == mqtt.ConnectReturnCode.ACCEPTED and not session_present:
        print("Session did not persist. Resubscribing to existing topics...")
        resubscribe_future, _ = connection.resubscribe_existing_topics()

        # Cannot synchronously wait for resubscribe result because we're on the connection's event-loop thread,
        # evaluate result with a callback instead.
        resubscribe_future.add_done_callback(on_resubscribe_complete)


def on_resubscribe_complete(resubscribe_future):
        resubscribe_results = resubscribe_future.result()
        print("Resubscribe results: {}".format(resubscribe_results))

        for topic, qos in resubscribe_results['topics']:
            if qos is None:
                sys.exit("Server rejected resubscribe to topic: {}".format(topic))


# Callback when the subscribed topic receives a message
def on_message_received(topic, payload, **kwargs):
    print("Received message from topic '{}': {}".format(topic, payload))
    # global received_count
    # received_count += 1
    # if received_count == args.count:
    #     received_all_event.set()

class AwsSink(Node):
    def __init__(self, endpoint=constants.AWS_ENDPOINT_JSON, cert=constants.AWS_CERT_PATH, key=constants.AWS_KEY_PATH, root_ca=constants.AWS_ROOT_CA_PATH, client_id='test-'+ str(uuid4()), topic='topic_1', feedback_address=gen_addr_ipc('feedback')):
        super().__init__('AwsSink')
        self.input_streams['feedback'] = InputStream(feedback_address)

        with open(endpoint, 'r') as f:
            j = json.load(f)
            self.endpoint = j['endpoint']
        self.cert = cert
        self.key = key
        self.root_ca = root_ca
        self.client_id = client_id
        self.topic = topic

    def prepare(self):
        self.received_count = 0
        self.received_all_event = threading.Event()

        self.event_loop_group = io.EventLoopGroup(1)
        self.host_resolver = io.DefaultHostResolver(self.event_loop_group)
        self.client_bootstrap = io.ClientBootstrap(self.event_loop_group, self.host_resolver)
        self.mqtt_connection = mqtt_connection_builder.mtls_from_path(
            endpoint=self.endpoint,
            cert_filepath=self.cert,
            pri_key_filepath=self.key,
            client_bootstrap=self.client_bootstrap,
            ca_filepath=self.root_ca,
            on_connection_interrupted=on_connection_interrupted,
            on_connection_resumed=on_connection_resumed,
            client_id=self.client_id,
            clean_session=False,
            keep_alive_secs=6)

        print("Connecting to {} with client ID '{}'...".format(
            self.endpoint, self.client_id))

        connect_future = self.mqtt_connection.connect()

        # Future.result() waits until a result is available
        print(connect_future.result())
        print("Connected!")

        # Subscribe
        print("Subscribing to topic '{}'...".format(self.topic))
        subscribe_future, packet_id = self.mqtt_connection.subscribe(
            topic=self.topic,
            qos=mqtt.QoS.AT_LEAST_ONCE,
            callback=on_message_received)

        subscribe_result = subscribe_future.result()
        print("Subscribed with {}".format(str(subscribe_result['qos'])))

    def process(self):
        feedback, sync = self.input_streams['feedback'].recv()
        if feedback is None:
            self.stop()
            return

        if 'feedback' not in feedback:
            return

        feedback['user_id'] = sync['user_id']
        feedback['set_id'] = sync['set_id']
        feedback['exercise'] = sync['exercise']
        feedback['timestamp'] = sync['timestamp']
        # print(feedback)

        print("Publishing message to topic '{}'".format(self.topic))
        ret = self.mqtt_connection.publish(
            topic=self.topic,
            payload=json.dumps(feedback),
            qos=mqtt.QoS.AT_LEAST_ONCE)
        print(ret)
         