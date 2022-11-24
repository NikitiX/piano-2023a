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

def fft(f):
    n = len(f)
    fft = []
    for k in range(n // 2):
        t = np.linspace(0, 2 * np.pi * k, n)
        intsin = np.sum(f * np.sin(t))
        a = 2 * intsin / n
        
        intcos = np.sum(f * np.cos(t))
        b = 2 * intcos / n
        fft += [complex(a,b)]
    return np.array(fft)

def ifft(fft, n):
    result = np.zeros(n)
    for k in range(n // 2):
        t = np.linspace(0, 2 * np.pi * k, n)
        result += np.sin(t) * np.real(fft[k])
        result += np.cos(t) * np.imag(fft[k])
    return result

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
    
    def __init__(self, fname = None, ws = 16,wch = 256):
        # ws - size of the window (precision of data) in samples
        # wch - amount of windows in a chunk (base block used for FFT)
        if fname is None:
            # "empty" case; used in load()
            return
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
        self.freq_map = None        # freq_map

        # File information
        self.length = len(self.data)                                        # length in samples
        self.time_length = self.length / self.samplerate                    # length in seconds
        self.wnd_amount = self.length // self.window_size                   # length in windows
        self.wnd_times = np.linspace(0, self.time_length, self.wnd_amount)  # starting time for each window
        
        self.process_file()

    def process_file(self):
        '''Processes the file, determining frequency components for each window.'''
        
        self.freqs = np.linspace(0, self.samplerate / 2, self.chunk_size // 2 + 1, endpoint=False)[1:]     # freq scale for FFT
        self.window_results = []    # FFT results
        self.chunk_results = [np.zeros(self.wnd_amount) for _ in range(self.chunk_size // 2)]    # FFT results
        self.phase = np.zeros(self.chunk_size // 2)
        counts = np.zeros(self.wnd_amount)
        z = np.zeros(self.chunk_size // 2).reshape((self.chunk_size // 2,1))
        
        prev_time = -9999999
        for offset in range(0,self.length - self.chunk_size,self.window_size): # Iterate through every chunk with the step of window_size
            chunk = self.data[offset : offset + self.chunk_size]

            # Progress report
            print(offset)
            time = offset / self.samplerate
            if time // 0.01 > prev_time // 0.01:
                print(round(time,2),'s')
            prev_time = time

            # Run the FFT
            yf = fft(chunk)

            # Save the results
            self.window_results.append(yf)
            wnd_offset = offset // self.window_size     # chunk index / first window index
            wnd_cnt = min(self.wnd_amount - wnd_offset, self.windows_in_chunk)
            tmp = np.concatenate(
                    [z] * wnd_offset + [np.abs(yf).reshape((self.chunk_size // 2,1))] * wnd_cnt + [z] * (self.wnd_amount - wnd_offset - wnd_cnt), axis=1)
            self.chunk_results += tmp
            tmp2 = np.array([0] * wnd_offset + [1] * wnd_cnt + [0] * (self.wnd_amount - wnd_offset - wnd_cnt))
            counts += tmp2

            # phase
            if offset == 0:
                for i in range(self.chunk_size // 2):
                    self.phase[i] = np.arctan2(np.imag(yf[i]), np.real(yf[i]))
    
        for i in range(self.chunk_size // 2):
            for j in range(self.wnd_amount):
                if counts[j] == 0:
                    self.chunk_results[i][j] = 0
                else:
                    self.chunk_results[i][j] /= counts[j]
        
        self.interpolate()

    def time_scale(self):
        return np.linspace(0, self.time_length, self.length)

    def interpolate(self):
        self.cr_interp = []
        for j in range(self.chunk_size // 2):
            #plt.plot(self.wnd_times,np.abs(self.chunk_results[j]))
            #plt.show()
            self.cr_interp.append(Interp(self.wnd_times, self.chunk_results[j]))

    def ifft(self):
        result = np.zeros(self.length)
        axis = self.time_scale()
        for k in range(self.chunk_size // 2):
            if k % 100 != 0 and 100 % k != 0:
                continue
            print(k)
            t = np.linspace(0, 2 * np.pi * k * self.length / self.chunk_size, self.length)
            tmp = np.sin(t + self.phase[k]) * self.cr_interp[k](axis)
            result += tmp
        return result

    def save(self, fname=None):
        if fname is None:
            fname = '.'.join(self.fname.split('.')[:-1]) + '.npy'
        print('saving to',fname,'...')
        with open(fname,'wb') as file:
            np.save(file, np.array([fname]))
            np.save(file, np.array([self.time_length]))
            np.save(file, np.array([self.samplerate,
                                          self.window_size,
                                          self.windows_in_chunk,
                                          self.chunk_size,
                                          self.length,
                                          self.wnd_amount]))
            np.save(file, np.array(self.data))
            np.save(file, self.wnd_times)
            np.save(file, self.freqs)
            np.save(file, np.array(self.window_results))
            np.save(file, np.array(self.chunk_results))
            np.save(file, np.array(self.phase))

    @staticmethod
    def load(fname):
        obj = FFT()
        with open(fname, 'rb') as file:
            obj.fname = np.load(file)[0]
            obj.time_length = np.load(file)[0]
            obj.samplerate, obj.window_size, obj.windows_in_chunk, obj.chunk_size, obj.length, obj.wnd_amount = np.load(file)
            obj.data = np.load(file)
            obj.wnd_times = np.load(file)
            obj.freqs = np.load(file)
            obj.window_results = np.load(file)
            obj.chunk_results = np.load(file)
            obj.phase = np.load(file)
        obj.interpolate()
        return obj

if __name__ == '__main__':
    '''
    ff = FFT('piano_short.wav', ws = 1000, wch = 10)

    xf = np.linspace(0, ff.time_length, ff.length)
    plt.plot(xf, ff.cr_interp[99](xf))
    plt.show()
    result = ff.ifft()
    if np.max(np.abs(result)) < 1:
        result *= 32767
    wavfile.write('processed.wav', ff.samplerate, result.astype(np.int16))
    '''
    
    ff = FFT.load('sin_waves2.npy')
    xf = np.linspace(0, ff.time_length, ff.length)
    result = ff.ifft()
    if np.max(np.abs(result)) < 1:
        result *= 32767
    xf = ff.time_scale()
    plt.plot(xf, result)
    plt.show()
    wavfile.write('processed.wav', ff.samplerate, result.astype(np.int16))
