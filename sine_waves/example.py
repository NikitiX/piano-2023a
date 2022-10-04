from generate_sin_waves import *

a = WavesGenerator()
a.add_freq()
a.add_freq(410, 10000)
a.add_freq(440, 5000, 1)
print(a)
x, y = a.generate()
print(y)
a.save(scale=0.5)
