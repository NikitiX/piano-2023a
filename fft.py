import numpy as np
from scipy.io.wavfile import read as read_wave
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

class FFT:
    '''Class for processing .wav files using Fast Fourier Transform.'''
    def __init__(self, fname, ws = 16,wch = 256):
        # ws - size of the window (precision of data) in samples
        # wch - amount of windows in a chunk (base block used for FFT)
        self.fname = fname
        self.samplerate,self.data = read_wave(fname)
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
        self.wnd_amount = self.length // self.window_size                   # length in windows
        self.wnd_times = np.linspace(0, self.time_length, self.wnd_amount)  # starting time for each window
        
        self.process_file()

    def process_file(self):
        '''Processes the file, determining frequency components for each window.'''
        
        self.freqs = rfftfreq(self.chunk_size, 1 / self.samplerate)     # freq scale for FFT
        self.window_results = [None for _ in range(self.wnd_amount)]    # FFT results
        counts = [0] * self.wnd_amount                                  # how many chunks intersect with each window (needed for calculating average)
        prev_time = 0.99

        for offset in range(0,self.length - self.chunk_size,self.window_size): # Iterate through every chunk with the step of window_size
            chunk = self.data[offset : offset + self.chunk_size]

            # Progress report
            time = offset / self.samplerate
            if time % 0.01 < prev_time % 0.01:
                print(round(time,2),'s')
            prev_time = time

            # Run the FFT
            yf = np.abs(rfft(chunk))

            # Save the results in respective windows
            # Each chunk covers (windows_in_chunk) windows, starting from its index
            wnd_offset = offset // self.window_size     # chunk index / first window index
            for j in range(wnd_offset, min(self.wnd_amount, wnd_offset + self.windows_in_chunk)):
                counts[j] += 1
                if self.window_results[j] is None:
                    self.window_results[j] = np.array(yf)
                else:
                    self.window_results[j] += yf

            # No future chunks will affect the window we just processed, so we can transform it into an Interp
            self.transform(wnd_offset, min(self.windows_in_chunk,wnd_offset + 1))

        # Transform the remaining windows
        for i in range(self.wnd_amount - self.windows_in_chunk + 1,self.wnd_amount):
            self.transform(i, counts[i])

    def transform(self, offset, count):
        '''Used inside process_file to turn an element of window_results into a more convenient to use structure
        np.array of frequency amplitudes ---> Interp(freq) = amplitude'''
        
        self.window_results[offset] = Interp(
            self.freqs,
            self.window_results[offset] / count)

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
        '''For each frequency, returns its function of amplitude from time.
        Not implemented yet.'''

        # Use cache
        if self.freq_map is not None:
            return self.freq_map

        # code goes here

def main():
    fname = '3_short.wav'

    fft = FFT(fname)

    
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
    

    time = 0.1
    print('time',time)
    xf = fft.get_freq_scale()
    func = fft.get_freq_function(time)
    yf = [func(i) for i in xf]
    plt.title('TIME: ' + str(round(time,2)) + ' s')
    plt.plot(xf,yf)
    plt.xlim(0,xf[-1])
    plt.ylim(0,max(yf) * 1.4)
    plt.show()

    
    
if __name__ == '__main__':
    main()
