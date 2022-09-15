import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.io.wavfile import read as read_wave
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d

class InterpXY:
    def __init__(self, xf, yf):
        self.xf = xf
        self.yf = yf
        self.f = interp1d(xf,yf,kind='cubic')
    def __call__(self,x):
        return self.f(x)

class FFT:
    window_size = 16
    windows_in_chunk = 256
    chunk_size = window_size * windows_in_chunk

    def __init__(self, fname):
        self.open(fname)

    def open(self,fname):
        self.fname = fname
        self.samplerate,self.data = read_wave(fname)
        self.interp_amps = {}
        self.process_file()

    def process_file(self):
        self.length = len(self.data)
        self.time_length = self.length / self.samplerate
        self.wnd_amount = self.length // FFT.window_size
        self.wnd_times = np.linspace(0, self.time_length, self.wnd_amount)
        self.freqs = rfftfreq(FFT.chunk_size, 1 / self.samplerate)
        self.ffts = []
        prev_time = 0.99
        for offset in range(0,len(self.data) - FFT.chunk_size,FFT.window_size):
            chunk = self.data[offset : offset + FFT.chunk_size]
            time = offset / self.samplerate
            if time % 0.1 < prev_time % 0.1:
                print(round(time,1),'s')
            prev_time = time
            yf = np.abs(rfft(chunk))
            self.ffts.append(InterpXY(self.freqs,yf))

    def create_amplitude_function(self,freq):
        sums = [0] * self.wnd_amount
        counts = [0] * self.wnd_amount
        prev_time = 0.99
        for i in range(self.wnd_amount - FFT.windows_in_chunk + 1):
            time = i * FFT.window_size / self.samplerate
            if time % 0.1 < prev_time % 0.1:
                print(round(time,1),'s')
            prev_time = time
            for j in range(FFT.windows_in_chunk):
                sums[i + j] += self.ffts[i](freq)
                counts[i + j] += 1
        ans = [0] * self.wnd_amount
        for i in range(self.wnd_amount):
            try:
                ans[i] = sums[i] / counts[i]
            except:
                ans[i] = 0
        return InterpXY(self.wnd_times,ans)

    def get_amplitude_function(self,freq):
        if freq in self.interp_amps:
            return self.interp_amps[freq]
        func = self.create_amplitude_function(freq)
        self.interp_amps[freq] = func
        return func

    def get_value(self,freq,time):
        return self.get_amplitude_function(freq)(time)

    def get_freqs(self,time):
        yf = []
        for freq in self.freqs:
            yf.push_back(self.get_value(freq,time))
        return InterpXY(self.freqs,yf)

def main():
    fname = '3.wav'

    fft = FFT(fname)
    while True:
        freq = float(input("ENTER FREQUENCY: "))
        func = fft.get_amplitude_function(freq)
        xf = np.linspace(0,fft.time_length,fft.wnd_amount)
        yf = func(xf)

        plt.title('FREQUENCY: ' + str(round(freq,2)) + ' Hz')
        plt.plot(xf,yf)
        plt.xlim(0,fft.time_length)
        plt.ylim(0,max(yf) * 1.4)
        plt.show()
    
if __name__ == '__main__':
    main()
