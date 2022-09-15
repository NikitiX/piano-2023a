import numpy as np #numpy
import csv
from matplotlib import pyplot as plt
from scipy.io.wavfile import read
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d

fname = '3.wav'
chunksize = 2 ** 12
windowsize = 2 ** 4

def fft(samplerate,data):
    times = [i * windowsize / samplerate for i in range(len(data) // windowsize)]
    ffts = [None] * (len(data) // windowsize)
    interp_ffts = [None] * (len(data) // windowsize)
    time_prev = 0
    for offset in range(0,len(data) - chunksize,windowsize):
        chunk = data[offset:offset + chunksize]
        time = offset / samplerate
        if time % 0.1 < time_prev % 0.1:
            print(round(time,1),'s')
        time_prev = time

        xf = rfftfreq(chunksize, 1 / samplerate)
        yf = np.abs(rfft(chunk))
        f = interp1d(xf,yf,kind='cubic')

        i = offset // windowsize
        ffts[i] = (xf,yf)
        interp_ffts[i] = f
    return times,ffts,interp_ffts

def func_freq(times,interp_ffts,freq):
    ans = [0] * (len(data) // windowsize)
    z = len(times) - (chunksize // windowsize)
    for i in range(z):
        
        if i % 1000 == 0:
            print(i,z)
        for j in range(chunksize // windowsize):
            ans[i + j] += interp_ffts[i](freq) / (chunksize // windowsize)
    for i in range(chunksize // windowsize):
        ans[i] = (ans[i] / (i + 1)) * (chunksize // windowsize)
        ans[-1-i] = (ans[-1-i] / (i + 1)) * (chunksize // windowsize)
    return ans

samplerate,data = read(fname)
print('Processing...')
times,ffts,interp_ffts = fft(samplerate,data)
print('Done!')
while True:
    freq = float(input('ENTER FREQUENCY: '))
    ans = func_freq(times,interp_ffts,freq)

    plt.title('FREQUENCY: ' + str(round(freq,2)) + ' Hz')
    plt.plot(times,ans)
    plt.ylim(0,max(ans) * 1.4)
    plt.xlim(0,times[-1])
    plt.show()
