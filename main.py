import os
import sys
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
import pyqtgraph as pg
import sounddevice as sd
from threading import Thread
import numpy as np
import math
from PyQt6.QtCore import QTimer
from scipy.signal import gaussian
from pydub import AudioSegment

from helpers.get_signal_from_file import get_signal_from_file
from models.signal import Signal
from enum import Enum
from functools import partial

mainwindow_ui_file_path = os.path.join(os.path.dirname(__file__), 'views', 'mainwindow.ui')
uiclass, baseclass = pg.Qt.loadUiType(mainwindow_ui_file_path)


class WindowType(Enum):
    RECTANGLE = 'rectangle'
    HAMMING = 'hamming'
    HANNING = 'hanning'
    GAUSSIAN = 'gaussian'

class ModeType(Enum):
    ANIMALS = 'animals'
    MUSIC = 'music'
    UNIFORM = 'uniform'
    ECG = 'ecg'

class MainWindow(uiclass, baseclass):
    
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Signal Equalizer Studio")
        self.signal = None
        self.output : Signal = None
        self.output_current_timer = QTimer(self)
        self.phase = None
        self.frequencies = None
        self.original_fourier_transform = None
        self.magnitude_dB = None
        self.fourier_transform = None
        self.current_timer = QTimer(self)
        self.window_type = WindowType.RECTANGLE
        self.mode = ModeType.ANIMALS
        self.slider_values = []
        self.lower_upper_freq_list = []
        self._initialize_signals_slots()

    def _initialize_signals_slots(self):
        self.import_action.triggered.connect(self._import_signal_file)
        self.input_play_button.pressed.connect(self.play_time_input)
        self.output_play_button.pressed.connect(self.play_time_output)
        self.input_slider.valueChanged.connect(lambda value: self._on_slider_change(value,isInput=True, signal= self.signal))
        self.output_slider.valueChanged.connect(lambda value: self._on_slider_change(value,isInput=False, signal= self.output))
        self.rectangle_button.pressed.connect(lambda: self.change_window(WindowType.RECTANGLE))
        self.gaussian_button.pressed.connect(lambda: self.change_window(WindowType.GAUSSIAN))
        self.hamming_button.pressed.connect(lambda: self.change_window(WindowType.HAMMING))
        self.hanning_button.pressed.connect(lambda: self.change_window(WindowType.HANNING))
        self.uniform_range_action.triggered.connect(lambda: self.change_mode(ModeType.UNIFORM))
        self.musical_instruments_action.triggered.connect(lambda: self.change_mode(ModeType.MUSIC))
        self.animal_sounds_action.triggered.connect(lambda: self.change_mode(ModeType.ANIMALS))
        self.ecg_abnormalities_action.triggered.connect(lambda: self.change_mode(ModeType.ECG))
        self.delete_action.triggered.connect(lambda: self.change_mode(self.mode))
        self.current_timer.timeout.connect(lambda: self.update_timer(isInput= True))
        self.output_current_timer.timeout.connect(lambda: self.update_timer(isInput= False))
        self.change_window(WindowType.RECTANGLE)
        self.change_mode(ModeType.ANIMALS)
        

    def delete_all(self):
        self.signal = None
        self.output : Signal = None
        self.frequencies = None
        self.original_fourier_transform = None
        self.fourier_transform = None
        self.slider_values = []
        self.magnitude_dB = None
        self.input_spectrogram_graph.canvas.axes.clear()
        self.input_spectrogram_graph.canvas.draw()
        self.output_spectrogram_graph.canvas.axes.clear()
        self.output_spectrogram_graph.canvas.draw()
        self.frequency_graph.clear()
        self.input_signal_graph.clear()
        self.output_signal_graph.clear()
        

    def _on_slider_change(self, value, isInput, signal):
        if signal:
            signal.current_time = value / 1000
            self.update_timer(isInput=isInput)

    def _import_signal_file(self):
        self.signal: Signal = get_signal_from_file(self)

        # plot time graph
        pen_c = pg.mkPen(color=(255, 255, 255))
        self.input_signal_graph.plot(self.signal.x_vec, self.signal.y_vec, pen=pen_c)
        self.input_signal_graph.setXRange(self.signal.x_vec[0], self.signal.x_vec[-1])
        self.input_signal_graph.setYRange(min(self.signal.y_vec), max(self.signal.y_vec))

        # plot initial output time graph
        # self.output_signal_graph.plot(self.signal.x_vec, self.signal.y_vec, pen=pen_c) 

        self.input_slider.setMinimum(0)
        self.input_slider.setMaximum(int(self.signal.x_vec[-1] * 1000))
        self.input_slider.setValue(0)
        self.input_total_time.setText(
            f'{str(math.floor(self.signal.x_vec[-1] / 60)).zfill(2)}:{str(math.floor(self.signal.x_vec[-1]) % 60).zfill(2)}')

        # plot input frequency graph
        self.plot_input_frequency()
        if self.mode == ModeType.UNIFORM:
            freq_list = []
            for i in range(10):
                lower_freq = i * (self.frequencies[-1] / 10)
                upper_freq = (i + 1) * (self.frequencies[-1] / 10)
                freq_list.append([lower_freq, upper_freq])
            self.lower_upper_freq_list = freq_list     
        
        self.plot_input_spectrograph()
        self.generate_output_signal()  
        self.perform_window()   

    def plot_input_frequency(self):

        self.frequencies, self.fourier_transform = self.apply_fourier_transform()
        self.original_fourier_transform = self.fourier_transform.copy()

        # Apply logarithmic transformation to y-axis values
        self.magnitude_dB = 20 * np.log10(abs(self.fourier_transform))

        # Plot the frequency graph
        pen_c = pg.mkPen(color=(255, 255, 255))
        self.frequency_graph.plot(self.frequencies, self.magnitude_dB, pen=pen_c)

        # Optionally, you can set labels for the axes
        self.frequency_graph.setLabel('left', 'Magnitude (dB)' )
        self.frequency_graph.setLabel('bottom', 'Frequency', units='Hz')

    def plot_spectrogram(self,canvas,signal,sample_rate,is_csv):
        canvas.axes.clear()
        canvas.axes.specgram(signal,Fs=sample_rate)

        canvas.draw()

    def plot_input_spectrograph(self):
        self.plot_spectrogram(
            canvas=self.input_spectrogram_graph.canvas,
            signal=self.signal.y_vec,
            sample_rate=self.signal.get_sampling_frequency(),
            is_csv= False
        )

    def plot_output_spectrograph(self):
        self.plot_spectrogram(
            canvas=self.output_spectrogram_graph.canvas,
            signal=self.output.y_vec,
            sample_rate=self.signal.get_sampling_frequency(),
            is_csv= False
        )


    def apply_fourier_transform(self):
        sampling_frequency = self.signal.audio.frame_rate if self.signal.audio else self.signal.get_sampling_frequency()

        fourier_transform = np.fft.rfft(self.signal.y_vec)
        frequencies = np.fft.rfftfreq(len(self.signal.y_vec), d= 1/sampling_frequency)
        self.phase = np.angle(fourier_transform)

        return frequencies, fourier_transform

        # Frequency domain representation
        amplitude = self.signal.y_vec
        
    

        fourier_transform = np.fft.rfft(amplitude)
        self.fourier_transform = np.abs(fourier_transform)
        tp_count = len(amplitude)

        # values = np.arange(int(tp_count))

        # time_period = tp_count / sampling_frequency

        # frequencies = values / time_period
        frequencies = np.fft.rfftfreq(tp_count, 1 / sampling_frequency)

        return frequencies, fourier_transform


    def play_time_input(self):
        if self.signal.audio:
            def is_playing_logic():
                sd.stop()
                self.signal.is_playing = False
                self.input_play_button.setText('Play')
                self.current_timer.stop()
            def is_not_playing_logic():
                self.audio_thread = Thread(target=lambda: self.play_audio(isInput=True))
                if self.signal.current_index == 0:
                    self.input_signal_graph.clear()
                self.audio_thread.start()
                self.current_timer.start(100)
                self.signal.is_playing = True
                self.input_play_button.setText('Pause')

            functions = {
                True: lambda: is_playing_logic(),
                False: lambda: is_not_playing_logic(),
            }    
            functions[self.signal.is_playing]()


    def slider_value_changed(self, index, value):
        self.slider_values[index].setText(f"{value/10}")
        if self.signal:
            self.perform_window()  

        
    def delete_sliders(self):  
      for i in reversed(range(self.sliders_layout.count())):
            item = self.sliders_layout.itemAt(i)
            if isinstance(item.layout(), QVBoxLayout):
                # Hide the widgets in the vertical layout
                for j in reversed(range(item.layout().count())):
                    widget = item.layout().itemAt(j).widget()
                    if widget:
                        widget.hide()
                self.sliders_layout.removeItem(item)
                item.layout().deleteLater()                
      

    def change_mode(self, mode_type):
        self.delete_all()
        if self.sliders_layout.count() != 0:
            self.delete_sliders()
        self.mode = mode_type
        def draw_sliders(label_list):
            for i in range(4):
                new_vertical_layout = QVBoxLayout()
                label = QLabel(label_list[i])
                slider = QSlider()
                slider.setRange(0,20)
                slider.setValue(10)
                value_label = QLabel('1')
                self.slider_values.append(value_label)
                new_vertical_layout.addWidget(label)
                new_vertical_layout.addWidget(slider)
                new_vertical_layout.addWidget(value_label)
                self.sliders_layout.addLayout(new_vertical_layout)
                slider.valueChanged.connect(partial(self.slider_value_changed, i))
        def animals_logic():
            freq_list = [
                [0,820],
                [820,2500],
                [2500,5000],
                [5000,11000],
             ]
            label_list = [ 'Owl', 'Horse', 'Songbird', 'Snake']
            self.lower_upper_freq_list = freq_list
            draw_sliders(label_list)

        def music_logic():
            freq_list = [
             [0,200],
            [200,1000],
            [1000,2000],
            [2000,5000],
        ]
            label_list = [ 'Bass', 'Piano', 'Trumpet', 'Drums']
            self.lower_upper_freq_list = freq_list
            draw_sliders(label_list)

        def ecg_logic():
            freq_list = [
            [0,3],
            [3, 20],
            [20,120],
            [120,180],
        ]
            label_list = [ 'SVTA', 'AFIB', 'WPW', 'Normal']
            self.lower_upper_freq_list = freq_list
            draw_sliders(label_list)

        def uniform_logic():
            for i in range(10):
                new_vertical_layout = QVBoxLayout()
                label = QLabel(f'Range {i+1}')
                slider = QSlider()
                slider.setRange(0,20)
                slider.setValue(10)
                value_label = QLabel('1')
                new_vertical_layout.addWidget(label)
                new_vertical_layout.addWidget(slider)
                new_vertical_layout.addWidget(value_label)
                self.slider_values.append(value_label)
                self.sliders_layout.addLayout(new_vertical_layout)
                slider.valueChanged.connect(partial(self.slider_value_changed, i))

        mode_dict = {
            ModeType.ANIMALS: lambda: animals_logic(),
            ModeType.MUSIC: lambda: music_logic(),
            ModeType.ECG: lambda: ecg_logic(),
            ModeType.UNIFORM: lambda: uniform_logic(),
        }    
        mode_dict[self.mode]()


    def change_window(self, window_type):
        self.window_type = window_type
        if self.signal is not None: 
            self.perform_window()
        self.gaussian_button.setStyleSheet("")
        self.hamming_button.setStyleSheet("")
        self.rectangle_button.setStyleSheet("")
        self.hanning_button.setStyleSheet("")
        style_sheet = "QPushButton { border: 2px solid #FFFFFF; }"  
        functions = {
            WindowType.GAUSSIAN: lambda : self.gaussian_button.setStyleSheet(style_sheet),
            WindowType.RECTANGLE: lambda : self.rectangle_button.setStyleSheet(style_sheet),
            WindowType.HAMMING: lambda : self.hamming_button.setStyleSheet(style_sheet),
            WindowType.HANNING: lambda : self.hanning_button.setStyleSheet(style_sheet),
        }
        functions[self.window_type]()


    def perform_window(self):
        total = len(self.slider_values)
        result = self.original_fourier_transform
        all_wave = np.array([])
        window_plot = np.array([])
        for i in range(total):
            lower_freq = self.lower_upper_freq_list[i][0]
            upper_freq = self.lower_upper_freq_list[i][1]
            amplitude = float(self.slider_values[i].text())
            freq_range_mask = self.frequencies[(self.frequencies >= lower_freq) & (self.frequencies <= upper_freq)]
            fourier_transform_mask = self.original_fourier_transform[(self.frequencies >= lower_freq) & (self.frequencies <= upper_freq)]
            functions = {
                WindowType.GAUSSIAN: lambda : gaussian(len(fourier_transform_mask), np.std(self.frequencies)) * amplitude,
                WindowType.RECTANGLE: lambda : np.ones(len(fourier_transform_mask)) * amplitude,
                WindowType.HAMMING: lambda : np.hamming(len(fourier_transform_mask)) * amplitude,
                WindowType.HANNING: lambda : np.hanning(len(fourier_transform_mask)) * amplitude,
            }
            signal = functions[self.window_type]()

            result = np.where(freq_range_mask, fourier_transform_mask * signal, fourier_transform_mask)
            all_wave = np.concatenate((all_wave, result))
            window_plot = np.concatenate((window_plot, signal))

        if len(all_wave) > len(self.fourier_transform):
            self.fourier_transform = all_wave[:len(self.fourier_transform)]
        else:    
            self.fourier_transform[:len(all_wave)] = all_wave
        self.frequency_graph.clear()
        self.frequency_graph.plot(self.frequencies, abs(self.original_fourier_transform.real))
        pen_c = pg.mkPen(color=(255, 0, 0))
        if len(window_plot) > len(self.frequencies):
            self.frequency_graph.plot(self.frequencies,window_plot[:len(self.frequencies)] * (max(self.original_fourier_transform.real)/10),pen= pen_c)
        else:  
            self.frequency_graph.plot(self.frequencies[:len(window_plot)],window_plot * (max(self.original_fourier_transform.real)/10),pen= pen_c)
        self.generate_output_signal()


    def generate_output_signal(self):
        self.output_signal_graph.clear()
        pen_c = pg.mkPen(color=(255, 255, 255))
        
        # Generate output using inverse Fourier transform of self.frequency and self.fourier_transform
        if self.fourier_transform is not None:

            # self.newSampleArr = np.fft.irfft(self.fourier_transform).real
            # y_vec = np.int16(self.newSampleArr) 
            data = (np.abs(self.fourier_transform) * np.exp(1j  * self.phase[:len(self.fourier_transform)]))
            y_vec = (np.fft.irfft(data).real) 
            # x_vec = self.signal.x_vec

            # y_vec = (np.fft.irfft((self.fourier_transform * np.exp(1j  * self.phase[:len(self.fourier_transform)]))).real) * -1
            x_vec = self.signal.x_vec[:len(y_vec)]
            self.output_signal_graph.plot(x_vec, y_vec, pen=pen_c)
            self.output_signal_graph.repaint()
            self.output_signal_graph.setXRange(x_vec[0], x_vec[-1])
            self.output_signal_graph.setYRange(min(y_vec), max(y_vec))
            if self.signal.audio is not None:
                y_vec = np.int16(y_vec)
                audio = AudioSegment(
                y_vec.astype(np.int16).tobytes(),
                frame_rate= None if self.signal.audio is None else self.signal.audio.frame_rate,
                sample_width=2,
                channels=1  
            )
                self.output = Signal(x_vec, y_vec, audio)

            else:
                self.output = Signal(x_vec, y_vec, audio= None)

            sd.stop()
            self.output.is_playing = False
            self.output_play_button.setText('Play')
            self.output_current_timer.stop()
            self.output_current_timer.stop()
            self.output.current_time = 0
            self.output.current_index = 0
            self.output_slider.setMinimum(0)
            self.output_slider.setMaximum(int(self.output.x_vec[-1] * 1000))
            self.output_slider.blockSignals(True)
            self.output_slider.setValue(0)
            self.output_slider.blockSignals(False)
            self.output_total_time.setText(
            f'{str(math.floor(self.output.x_vec[-1] / 60)).zfill(2)}:{str(math.floor(self.output.x_vec[-1]) % 60).zfill(2)}')
            self.plot_output_spectrograph()


    def play_audio(self, isInput):
        signal = self.signal if isInput else self.output
        def input_logic():
            self.signal.is_playing = True
            return self.input_play_button
        def output_logic():
             self.output.is_playing = True
             return self.output_play_button
        
        fun_dict = {
            True: lambda: input_logic(),
            False: lambda: output_logic(),
        }
        
        if signal.audio:
            button = fun_dict[isInput]()
            final_index = np.abs(signal.x_vec - signal.current_time).argmin()
            sd.play(signal.y_vec[final_index:], signal.audio.frame_rate * 2)
            sd.wait()
            self.current_timer.stop() if isInput else self.output_current_timer.stop()
            signal.is_playing = False
            
            if final_index >= len(signal.x_vec) - 100:
                button.setText('Rewind')
                signal.current_time = 0
                signal.current_index = 0



    def play_time_output(self):
        if self.output is not None and self.signal.audio:

            def is_playing_logic():
                sd.stop()
                self.output.is_playing = False
                self.output_play_button.setText('Play')
                self.output_current_timer.stop()

            def is_not_playing_logic():
                self.output_audio_thread = Thread(target=lambda: self.play_audio(isInput=False))
                if self.output.current_index == 0:
                    self.output_signal_graph.clear()
                self.output_audio_thread.start()
                self.output_current_timer.start(100)
                self.output.is_playing = True
                self.output_play_button.setText('Pause')

            functions = {
                True: lambda: is_playing_logic(),
                False: lambda: is_not_playing_logic(),
            }    
            functions[self.output.is_playing]()


    def update_timer(self, isInput):
        def is_input_logic():
            signal = self.signal
            graph = self.input_signal_graph
            return signal, graph
        
        def is_output_logic():
            signal = self.output
            graph = self.output_signal_graph
            return signal, graph
        
        functions = {
            True: lambda: is_input_logic(),
            False: lambda: is_output_logic(),
        }
        signal, graph = functions[isInput]()

        signal.current_time += 0.1    
        current_text = self.current_input_time if isInput else self.current_output_time
        current_slider = self.input_slider if isInput else self.output_slider

        current_text.setText(
            f'{str(math.floor(signal.current_time / 60)).zfill(2)}:{str(math.floor(signal.current_time) % 60).zfill(2)}')
        current_slider.blockSignals(True)
        current_slider.setValue(math.ceil(signal.current_time * 1000))
        current_slider.blockSignals(False)
        current_slider.repaint()
        old_current_output_index = signal.current_index
        signal.current_index += math.ceil(len(signal.x_vec) / (signal.x_vec[-1] * 10))
        graph.plot(signal.x_vec[old_current_output_index:signal.current_index], signal.y_vec[old_current_output_index:signal.current_index])



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
