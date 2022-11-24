import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from pprint import pprint

class Interp:
    '''A basic class with interpolation/dictionary functionality.'''
    def __init__(self, xf = [], yf = []):
        self.xf = list(xf)
        self.yf = list(yf)
        #print(len(self.xf), len(self.yf))
        self.d = dict(zip(self.xf, self.yf))
        self.f = interp1d(self.xf,
                          self.yf,
                          kind='cubic', fill_value="extrapolate")
    def value(self,x):
        return self.d[x] if x in self.d else self.f(x)
    def __call__(self,x):
        if type(x) == np.ndarray:
            return [self.value(i) for i in x]
        return self.value(x)

length = 5 * 44100
time_length = 5
chunk_size = 10000

xf = np.linspace(0, time_length, length)
cr_interp = [
    Interp(xf, 0.5 * np.exp(-xf * 0.5)),
    Interp(xf, 0.3 * np.exp(-xf * 0.7))]
phase = [0,0]

result = np.zeros(length)
axis = np.linspace(0, time_length, length)
for k in range(len(cr_interp)):
    print(k)
    t = np.linspace(0, 200 * np.pi * (k + 1) * length / chunk_size, length)
    tmp = np.sin(t + phase[k]) * cr_interp[k](axis)
    result += tmp
if np.max(np.abs(result)) < 1:
    result *= 32767
wavfile.write('proc3.wav', 44100, result.astype(np.int16))
