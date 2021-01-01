import math
import numpy as np
import lib6003
import matplotlib.pyplot as plt
from lib6003.audio import wav_write
from lib6003.audio import wav_file_play
from lib6003.audio import wav_read
from lib6003.audio import wav_play
from lib6003.fft import fft
from lib6003.fft import ifft
from scipy.io.wavfile import write

# wav_file_play('ocean_man_raw.wav')
# wav_file_play('ocean_man.wav')
data, samp_rate = wav_read('ocean_man.wav')
N = len(data)
# print(N)
dft_cof = fft(data)
k = np.linspace(0,N-1,N)
f=k*samp_rate/N
plt.plot(f,np.absolute(dft_cof))
plt.xlim(850,1000)
plt.ylim(0,0.001)
plt.show()
k_cut = int((990*N/samp_rate))
# print(k_cut)
freqs = fft(data)
# print(len(freqs))
for i in range(k_cut-100,k_cut+100):
    freqs[i]= 0
for i in range(-k_cut-100,-k_cut+100):
    freqs[i]= 0
# print(len(freqs))
f=k*samp_rate/N

# plt.plot(f,np.absolute(freqs))
# plt.show()
# ocean_fix = ifft(freqs)
# wav_play(ocean_fix,samp_rate)
