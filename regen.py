from fft import FFT
from sine_waves.generate_sin_waves import WavesGenerator
import numpy as np
from scipy.io import wavfile

fname = 'sin_waves.wav'
fft = FFT(fname)
fmap = fft.get_freq_map()
result = np.zeros(fft.length)
xf = np.linspace(0, fft.time_length, fft.length, endpoint=False)
for freq in fmap:
    if freq > 1000:
        break
    print(freq)
    res = (np.sin(2 * np.pi * xf * freq)) * fmap[freq](xf)
    print(res)
    result += res
norm = max(np.min(result) / -32768,np.max(result) / 32767)
result /= norm
print(np.min(result),np.max(result))
wavfile.write('output.wav', fft.samplerate, result.astype(np.int16))
