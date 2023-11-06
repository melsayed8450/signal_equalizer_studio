from enum import Enum

class SignalType(Enum):
    CONTINUOUS = 0
    DISCRETE = 1

class Signal:
    def __init__(self, x_vec, y_vec, audio= None, signal_type: SignalType = SignalType.CONTINUOUS) -> None:
        self.x_vec = x_vec
        self.y_vec = y_vec
        self.audio = audio
        self.signal_type = signal_type