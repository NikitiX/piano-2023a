'''An examples file for FFT.py.

Does not cover:
1) test/comparison between on-scale/arbitrary freqs/times
2) freq_map
3) file updating with different ws/wch
'''

from fft import FFT
from sine_waves.generate_sin_waves import WavesGenerator
from matplotlib import pyplot as plt
import numpy as np

def read_from_file():
    return FFT('95326__ramas26__a.wav')

def read_from_sines():
    '''
    gen = WavesGenerator()
    gen.add_freq(100, 10000)
    gen.add_freq(200, 20000)
    gen.add_freq(500, 5000)
    gen.add_freq(1000, 7500)
    gen.add_freq(2000, 3750)
    gen.save(duration=1, scale=0.5)
    '''
    return FFT('sin_waves.wav')


def amp_func_test(fft, freq):
    print('freq',freq)
    xf = fft.get_time_scale()
    func = fft.get_amplitude_function(freq)
    yf = func(xf)
    plt.title('FREQUENCY: ' + str(round(freq,2)) + ' Hz')
    plt.plot(xf,yf)
    plt.xlim(0,fft.time_length)
    plt.ylim(0,max(yf) * 1.4)
    plt.show()

def freq_func_test(fft, time):
    print('time',time)
    xf = fft.get_freq_scale()
    func = fft.get_freq_function(time)
    yf = func(xf)
    plt.title('TIME: ' + str(round(time,2)) + ' s')
    plt.plot(xf,yf)
    plt.xlim(0,xf[-1])
    plt.ylim(0,max(yf) * 1.4)
    plt.show()

def freq_map_test(fft, freq):
    print('map_freq',freq)
    xf = fft.get_time_scale()
    fmap = fft.get_freq_map()
    func = fmap[freq]
    yf = func(xf)
    plt.title('FREQUENCY: ' + str(round(freq,2)) + ' Hz')
    plt.plot(xf,yf)
    plt.xlim(0,fft.time_length)
    plt.ylim(0,max(yf) * 1.4)
    plt.show()

fft = read_from_sines()
freq_func_test(fft, 0.5)
'''
xf = fft.get_time_scale()
for freq in {100, 200, 500, 1000, 2000, 150, 50, 505, 1001, 1002}:
    yf = fft.get_amplitude_function(freq)(xf)
    print(freq, np.average(yf))
    plt.title('FREQUENCY: ' + str(round(freq,2)) + ' Hz')
    plt.plot(xf,yf)
    plt.xlim(0,fft.time_length)
    plt.ylim(0,max(yf) * 1.4)
    plt.show()
'''
