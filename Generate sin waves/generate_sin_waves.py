import numpy as np
from scipy.io import wavfile
import math

class WavesGenerator:
    """Class for generating sin waves and writing to .wav file"""
    def __init__(self):
        self.frequencies = []  # list of frequencies, amplitudes, phases
                               # [(freq, amp, phase), (440, 100, 0), ]

    def add_freq(self, freq=440, amplitude=30000, phase=0):
        self.frequencies.append((freq, amplitude, phase))

    def remove_freq(self, freq=None, amplitude=None, phase=None):
        """Remove data from frequencies list
        If data in self.frequencies list matches the pattern in input than it is removed.
        """
        self.frequencies = list(filter(lambda x: ((x[0] != freq and freq is not None) or (x[1] != amplitude and amplitude is not None) or (x[2] != phase and phase is not None)), self.frequencies))

    def generate(self, sample_rate=44100, duration=5):
        """Generate sum of sin waves from self.frequencies data

        Parameters
        ----------
        sample_rate : int
            The number of samples per second.
        duration : int
            Duration of signal. In seconds.

        Returns
        -------
        x : numpy.ndarray
            Array of time stamps for each sample.
        y : numpy.ndarray
            Array of samples.
        """
        x = np.linspace(0, duration, sample_rate*duration, endpoint=False)
        y = np.zeros(sample_rate*duration)
        for i in range(len(y)):
            for item in self.frequencies:
                y[i] += item[1] * math.sin((x[i] * 2 * np.pi + item[2]) * item[0])
                # A * sin((2*pi * x + phase) * freq)
        return x, y

    def save(self, file_name="sin_waves.wav", sample_rate=44100, duration=5, scale=1):
        """Generate sum of sin waves and save to file

        Parameters
        ----------
        file_name : str
            Name of target file.
        sample_rate : int
            The number of samples per second.
        duration : int
            Duration of signal. In seconds.
        scale : float
            Scale the final signal by this value.
        """
        x, y = self.generate(sample_rate, duration)
        y *= scale
        cuted_flag = False
        for i in range(len(y)):
            if y[i] > 32767:
                y[i] = 32767
                cuted_flag = True
            if y[i] < -32768:
                y[i] = -32768
                cuted_flag = True
        if cuted_flag:
            print("!!! The signal is cut off. Values are out of bounds (-32768, 32767)")
        wavfile.write(file_name, sample_rate, y.astype(np.int16))

    def __str__(self):
        ret = "#### Freq list ####\n"
        for item in self.frequencies:
            ret += f'{item[0]} Гц; amp={item[1]}; phase={item[2]}\n'
        ret += "###################"
        return ret
