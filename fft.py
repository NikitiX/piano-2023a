import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d

class Interp:
    '''A basic class with interpolation/dictionary functionality.'''
    def __init__(self, xf = [], yf = []):
        self.xf = list(xf)
        self.yf = list(yf)
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

def fft(f):
    n = len(f)
    f[0] *= 0.5
    f[-1] *= 0.5
    fft = []
    for k in range(n // 2):
        t = np.linspace(0, 2 * np.pi * k, n)
        intsin = np.sum(f * np.sin(t))
        a = 2 * intsin / n
        
        intcos = np.sum(f * np.cos(t))
        b = -2 * intcos / n
        fft += [complex(a,b)]
    return np.array(fft)

def ifft(fft, n):
    result = np.zeros(n)
    for k in range(n // 2):
        t = np.linspace(0, 2 * np.pi * k, n)
        result += np.sin(t) * np.real(fft[k])
        result += np.cos(t) * np.imag(fft[k])
    return -np.flip(result)

def mask(n, length, exclude_zero=False):
    exclude_zero = int(exclude_zero)
    x = np.linspace(exclude_zero, n // 2 + exclude_zero, n // 2, endpoint=False)
    x_rev = np.flip(x)
    x_new = np.concatenate([x, x_rev])
    x_new[x_new > length] = length
    x_new[x_new > n - length] = n - length
    return x_new / length

class FFT:
    '''Class for processing .wav files using Fast Fourier Transform.'''
    def __init__(self, fname, ws = 16,wch = 256):
        # ws - size of the window (precision of data) in samples
        # wch - amount of windows in a chunk (base block used for FFT)
        self.fname = fname
        self.samplerate,self.data = wavfile.read(fname)
        self.process(ws, wch)

    def process(self, ws = 16, wch = 256):
        '''Initialize the class for file processing.'''

        # Constants
        self.window_size = ws
        self.windows_in_chunk = wch
        self.chunk_size = ws * wch

        # Caches for data retrieval functions
        self.interp_amps = {}       # amplitude_function
        self.interp_freqs = {}      # freq_function
        self.freq_map = None        # freq_map (not implemeted yet)

        # File information
        self.length = len(self.data)                                        # length in samples
        self.time_length = self.length / self.samplerate                    # length in seconds
        
        self.process_file()

    def process_file(self):
        '''Processes the file, determining frequency components for each window.'''
        
        self.freqs = np.linspace(0, self.samplerate / 2, self.chunk_size // 2, endpoint=False)[1:]     # freq scale for FFT
        self.window_results = []    # FFT results
        prev_time = 0.99

        for offset in range(0,self.length - self.chunk_size,self.window_size): # Iterate through every chunk with the step of window_size
            chunk = self.data[offset : offset + self.chunk_size]

            # Progress report
            print(offset)
            time = offset / self.samplerate
            if time % 0.01 < prev_time % 0.01:
                print(round(time,2),'s')
            prev_time = time

            # Run the FFT
            yf = fft(chunk)

            # Save the results
            self.window_results.append(yf)
        
        self.wnd_amount = len(self.window_results)                                                              # length in windows
        self.wnd_times = np.linspace(0, self.wnd_amount * self.window_size / self.samplerate, self.wnd_amount)  # starting time for each window

    def ifft(self):
        result = np.zeros(self.length)
        m = mask(self.chunk_size, self.window_size) / (self.windows_in_chunk - 1)
        print(m)
        for i in range(self.wnd_amount):
            print(i * self.window_size)
            window_ifft = ifft(self.window_results[i], self.chunk_size) * m
            tmp = np.concatenate([
                np.zeros(i * self.window_size),
                window_ifft,
                np.zeros(self.length - i * self.window_size - self.chunk_size)])
            result += tmp
        result /= mask(self.length, self.chunk_size, True)
        print(list(result))
        return result / np.max(np.abs(result)) * 32767

    def get_freq_scale(self):
        return self.freqs

    def get_time_scale(self):
        return self.wnd_times

    def get_value(self, freq, time):
        '''Returns the amplitude of an arbitrary frequency in an arbitrary moment of time.'''

        '''
                window
        [-------------------------]
        ^         ^               ^
        |       time              |
        |                         |
    first_index          first_index + 1
    first_time           second_time

        To get the value within the window, we use linear interpolation using the values on the edges (this window and next window).
        '''
        
        first_index = int(time // (self.window_size / self.samplerate))
        first_time = first_index * self.window_size / self.samplerate
        second_time = first_time + self.window_size / self.samplerate
        count = (time - first_time) / (second_time - first_time)

        # Using min to prevent the out of bounds exception
        # The Interp classes in window_results cover the problem of getting an arbitrary frequency
        res_1 = self.window_results[min(first_index, self.wnd_amount - 1)](freq)
        res_2 = self.window_results[min(first_index + 1, self.wnd_amount - 1)](freq)
        res = count * res_1 + (1 - count) * res_2
        return res

    def get_amplitude_function(self,freq):
        '''Returns the function of amplitude from time for an arbitrary frequency.'''

        # Use cache
        if freq in self.interp_amps:
            return self.interp_amps[freq]

        ans = []     # amplitude for each window
        for time in self.wnd_times:
            # Fetch the answer using get_value
            ans.append(self.get_value(freq, time))
        
        func = Interp(self.wnd_times, ans)
        self.interp_amps[freq] = func       # save the result to cache
        return func

    def get_freq_function(self,time):
        '''Returns the function of amplitude from frequency for an arbitrary time.'''

        # Use cache
        if time in self.interp_freqs:
            return self.interp_freqs[time]
        
        ans = []    # amplitude for each frequency
        for freq in self.freqs:
            # Fetch the answer using get_value
            ans.append(self.get_value(freq,time))

        func = Interp(self.freqs,ans)
        self.interp_freqs[time] = func       # save the result to cache
        return func

    def get_freq_map(self):
        '''For each frequency, returns its function of amplitude from time.'''

        # Use cache
        if self.freq_map is not None:
            return self.freq_map

        d = {}
        prev_freq = 999
        for freq in self.freqs:
            # Progress report
            if freq % 1000 < prev_freq % 1000:
                print(freq,'Hz')
            prev_freq = freq
            
            d[freq] = self.get_amplitude_function(freq)
            
        self.freq_map = d
        return d

if __name__ == '__main__':
    ff = FFT('test_file.wav')
    result = ff.ifft()
    print(result)
    wavfile.write('processed.wav', ff.samplerate, result.astype(np.int16))
