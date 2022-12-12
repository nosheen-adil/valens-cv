import zmq
from enum import Enum

class State(Enum):
    IDLE = 'idle'
    RUNNING = 'running'

class Signal(Enum):
    START = 'start'
    STOP = 'stop'

def main():
    address = 'tcp://*:5000'

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(address)
    print('Bound to:', address)
    
    state = State.IDLE
    while True:
        req = socket.recv_json()
        if 'signal' not in req:
            print('Error: no signal')
            rep = {'success' : False, 'error' : 'bad request'}
            socket.send_json(rep)
            continue

        signal = Signal(req['signal'])
        print('Received request:', req)

        if signal is Signal.START:
            if state is State.IDLE:
                print('Starting...')
                state = State.RUNNING
                rep = {
                    'success' : True,
                    'state' : state.value
                }
                socket.send_json(rep)
            else:
                assert state is State.RUNNING
                print('Error: received start when running, stopping now')
                state = State.IDLE
                rep = {
                    'success' : False,
                    'error' : 'already running',
                    'state' : state.value
                }
                socket.send_json(rep)
        else:
            assert signal is Signal.STOP
            if state is State.IDLE:
                print('Error: received stop but not running')
                rep = {
                    'success' : False,
                    'error' : 'not running',
                    'state' : state.value
                }
                socket.send_json(rep)
            else:
                assert state is State.RUNNING
                print('Stopping...')
                state = State.IDLE
                rep = {
                    'success' : True,
                    'state' : state.value
                }
                socket.send_json(rep)

if __name__ == '__main__':
    main()
