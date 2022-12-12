from valens.signal import Signal, SignalSender, SignalReceiver

def test_signal_send_recv():
    sender = SignalSender()
    receiver = SignalReceiver()

    sender.start()
    receiver.start()

    sender.send(Signal.start)
    signal = receiver.recv()
    assert signal is Signal.start

    sender.send(Signal.reset)
    signal = receiver.recv()
    assert signal is Signal.reset

    sender.send(Signal.stop)
    signal = receiver.recv()
    assert signal is Signal.stop

def test_signal_recv_none():
    sender = SignalSender()
    receiver = SignalReceiver()

    sender.start()
    receiver.start()

    signal = receiver.recv(timeout=10)
    assert signal is None
