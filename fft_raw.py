import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq

samplerate, f = wavfile.read('sin_waves.wav')
n = len(f)
f[0] *= 0.5
f[-1] * 0.5
f_rec = np.zeros(n)
fft = []
for k in range(n // 2):
    if k % 100 == 0:
        print('k:',k)
    t = np.linspace(0, 2 * np.pi * k, n)
    intsin = np.sum(f * np.sin(t))
    a = 2 * intsin / n
    f_rec += a * np.sin(t)
    
    intcos = np.sum(f * np.cos(t))
    b = -2 * intcos / n
    f_rec += b * np.cos(t)
    fft += [complex(a,b)]
print(np.max(np.abs(f)))
print(f_rec)
print(np.array(fft))
wavfile.write('processed.wav', samplerate, f_rec.astype(np.int16))
