import numpy as np
import csv
import pprint
from matplotlib import pyplot as plt
from scipy.io.wavfile import read as read_wave
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d

class Interp:
    def __init__(self, xf = [], yf = []):
        self.xf = list(xf)
        self.yf = list(yf)
        self.d = dict(zip(self.xf, self.yf))
        self.f = interp1d(self.xf,
                          self.yf,
                          kind='cubic', fill_value="extrapolate")
    def __call__(self,x):
        return self.d[x] if x in self.d else self.f(x)

class FFT:
    def __init__(self, fname, ws = 16,wch = 256):
        self.fname = fname
        self.samplerate,self.data = read_wave(fname)
        self.freq_map = None
        self.reprocess(ws, wch)

    def reprocess(self, ws = 16, wch = 256):
        self.window_size = ws
        self.windows_in_chunk = wch
        self.chunk_size = ws * wch
        self.process_file()

    def process_file(self):
        self.interp_amps = {}
        self.interp_freqs = {}
        self.length = len(self.data)
        self.time_length = self.length / self.samplerate
        self.wnd_amount = self.length // self.window_size
        self.wnd_times = np.linspace(0, self.time_length, self.wnd_amount)
        self.freqs = rfftfreq(self.chunk_size, 1 / self.samplerate)
        self.ffts = []
        self.window_results = [None for _ in range(self.wnd_amount)]
        counts = {}
        prev_time = 0.99
        for offset in range(0,self.length - self.chunk_size,self.window_size):
            chunk = self.data[offset : offset + self.chunk_size]
            time = offset / self.samplerate
            if time % 0.01 < prev_time % 0.01:
                print(round(time,2),'s')
            prev_time = time
            yf = np.abs(rfft(chunk))
            self.ffts.append(Interp(self.freqs,yf))
            wnd_offset = offset // self.window_size
            for j in range(wnd_offset,min(self.wnd_amount,wnd_offset + self.windows_in_chunk)):
                counts.setdefault(j,0)
                counts[j] += 1
                if self.window_results[j] is None:
                    self.window_results[j] = np.array(yf)
                else:
                    self.window_results[j] += yf
            self.window_results[wnd_offset] = Interp(
                    self.freqs,
                    self.window_results[wnd_offset] / min(self.windows_in_chunk,wnd_offset + 1))
        #pprint.pprint(counts)
        for i in range(self.wnd_amount - self.windows_in_chunk + 1,self.wnd_amount):
            #print(i,self.wnd_amount)
            self.window_results[i] = Interp(
                self.freqs,
                self.window_results[i] / counts[i])
        self.chunk_amount = len(self.ffts)

    def get_freq_scale(self):
        return self.freqs

    def get_time_scale(self):
        return self.wnd_times

    def get_value(self, freq, time):
        first_index = int(time // (self.window_size / self.samplerate))
        first_time = first_index * self.window_size / self.samplerate
        second_time = first_time + self.window_size / self.samplerate
        count = (time - first_time) / (second_time - first_time)
        res_1 = self.window_results[min(first_index,self.wnd_amount - 1)](freq)
        res_2 = self.window_results[min(first_index + 1,self.wnd_amount - 1)](freq)
        res = count * res_1 + (1 - count) * res_2
        '''
        print(freq,
              time,
              first_index,
              round(res_1,2),
              round(res_2,2),
              round(res,2))
        '''
        return res

    def create_amplitude_function(self,freq):
        ans = [0] * self.wnd_amount
        prev_time = 0.99
        for i in range(self.wnd_amount):
            if self.wnd_times[i] % 0.1 < prev_time % 0.1:
                print(round(self.wnd_times[i],1),'s')
            prev_time = self.wnd_times[i]
            ans[i] = self.get_value(freq, self.wnd_times[i])
        return Interp(self.wnd_times,ans)

    def get_amplitude_function(self,freq):
        if freq in self.interp_amps:
            return self.interp_amps[freq]
        func = self.create_amplitude_function(freq)
        self.interp_amps[freq] = func
        return func

    def create_freq_map(self,time):
        yf = []
        for freq in self.freqs:
            yf.append(self.get_value(freq,time))
        return Interp(self.freqs,yf)

    def get_freq_map(self,time):
        if time in self.interp_freqs:
            return self.interp_freqs[time]
        func = self.create_freq_map(time)
        self.interp_freqs[time] = func
        return func

def main():
    fname = '3_short.wav'

    fft = FFT(fname)

    '''
    freq = 100
    xf = fft.get_time_scale()
    print('freq',freq)
    func = fft.get_amplitude_function(freq)
    yf = [func(i) for i in xf]
    plt.title('FREQUENCY: ' + str(round(freq,2)) + ' Hz')
    plt.plot(xf,yf)
    plt.xlim(0,fft.time_length)
    plt.ylim(0,max(yf) * 1.4)
    plt.show()
    '''

    time = 0.1
    print('time',time)
    xf = fft.get_freq_scale()
    func = fft.get_freq_map(time)
    yf = [func(i) for i in xf]
    plt.title('TIME: ' + str(round(time,2)) + ' s')
    plt.plot(xf,yf)
    plt.xlim(0,xf[-1])
    plt.ylim(0,max(yf) * 1.4)
    plt.show()

    
    
if __name__ == '__main__':
    main()
