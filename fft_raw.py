import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq

f = np.array([0,1,0,-1,0])
n = len(f) - 1
f_rec = np.zeros(n + 1)
fft = []
for k in range(n):
    print('k:',k)
    t = np.linspace(0, 2 * np.pi * k, n + 1)
    intsin = np.sum(f * np.sin(t))
    a = intsin / n
    f_rec += a * np.sin(t)
    
    intcos = np.sum(f * np.cos(t))
    b = intcos / n
    f_rec += b * np.cos(t)
    fft += [complex(a,b)]
print(f)
print(f_rec)
print(fft)
print([i.imag for i in fft])
print([i.real for i in fft])
f_rec *= 32768
#wavfile.write('processed.wav', samplerate, f_rec.astype(np.int16))
