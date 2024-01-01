from abc import ABC, abstractmethod
import pandas as pd
from models.signal import Signal
from pydub import AudioSegment
import numpy as np

# Interface that describes how signal loaders should be implemented
class ISignalLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> Signal:
        pass


class TextSignalLoader(ISignalLoader):
    def load(self, file_path: str) -> Signal:
        data = pd.read_csv(file_path)
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
        return Signal(x, y)
    
class CSVSignalLoader(ISignalLoader):
    def load(self, file_path: str) -> Signal:
        data = pd.read_csv(file_path)
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
        print(len(y))
        return Signal(x, y)

class ExcelXSignalLoader(ISignalLoader):
    def load(self, file_path: str) -> Signal:
        data = pd.read_excel(file_path)
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
        return Signal(x, y)
    
class ExcelSignalLoader(ISignalLoader):
    def load(self, file_path: str) -> Signal:
        data = pd.read_excel(file_path)
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
        return Signal(x, y)
    
class Mp3SignalLoader(ISignalLoader):
    def load(self, file_path: str) -> Signal:
        audio = AudioSegment.from_mp3(file_path)
        audio_data = np.array(audio.get_array_of_samples())
        time = np.arange(0, len(audio_data)) / (audio.frame_rate * 2)
        return Signal(time, audio_data, audio)    
    
class WavSignalLoader(ISignalLoader):
    def load(self, file_path: str) -> Signal:
        audio = AudioSegment.from_wav(file_path)
        audio_data = np.array(audio.get_array_of_samples())
        time = np.arange(0, len(audio_data)) / (audio.frame_rate )
        return Signal(time, audio_data, audio)    