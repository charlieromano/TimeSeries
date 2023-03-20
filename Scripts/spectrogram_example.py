import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

time_step = .01
time_vec = np.arange(0, 70, time_step)

# A signal with a small frequency chirp
sig = np.sin(0.5 * np.pi * time_vec * (1 + .1 * time_vec))

plt.figure(figsize=(8, 5))
plt.plot(time_vec, sig)




import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

inputfile = "../Datasets/TECO2.2010.2021.csv"
df = pd.read_csv(inputfile)
df=df[['fechaHora','ultimoPrecio',]]
df=df.set_index(pd.DatetimeIndex(df['fechaHora']))
df.plot()
plt.title("timeseries: "+inputfile)
plt.xlabel("time")
plt.ylabel("AR$ ")
plt.grid()



from scipy import signal

sig = df.ultimoPrecio

freqs, times, spectrogram = signal.spectrogram(sig,1e3)

#plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
plt.pcolormesh(times, freqs, spectrogram, shading='gouraud')
plt.title('Spectrogram')
plt.ylabel('Frequency band')
plt.xlabel('Time window')
plt.tight_layout()
plt.show()


fs = 1e3

# Ejemplo 2
inputfile = "../Datasets/S1MME_week43.csv"
df = pd.read_csv(inputfile)
sig=df.S1_mode_combined_attach_request_times_SEQ
freqs, times, spectrogram = signal.spectrogram(sig,fs)
plt.pcolormesh(times, freqs, spectrogram, shading='gouraud')
plt.title('Spectrogram')
plt.ylabel('Frequency band')
plt.xlabel('Time window')
plt.tight_layout()
plt.show()



# Ejemplo 3
inputfile = "../Datasets/TECO2.2000.2021.csv"
df = pd.read_csv(inputfile)
sig = df.ultimoPrecio
freqs, times, spectrogram = signal.spectrogram(sig,fs)
plt.pcolormesh(times, freqs, spectrogram, shading='gouraud')
plt.title('Spectrogram')
plt.ylabel('Frequency band')
plt.xlabel('Time window')
plt.tight_layout()
plt.show()


f, Pxx_den = signal.periodogram(sig, fs)
plt.semilogy(f, Pxx_den)
plt.ylim([1e-7, 1e2])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()


# Ejemplo 4
inputfile = "../Datasets/BYMA.csv"
df = pd.read_csv(inputfile)
sig = df.ultimoPrecio
freqs, times, spectrogram = signal.spectrogram(sig,5)
plt.pcolormesh(times, freqs, spectrogram, shading='gouraud')
plt.title('Spectrogram')
plt.ylabel('Frequency band')
plt.xlabel('Time window')
plt.tight_layout()
plt.show()


inputfile = "../Datasets/S1MME_week43.csv"
df = pd.read_csv(inputfile)
sig=df.S1_mode_combined_attach_request_times_SEQ

inputfile = "../Datasets/TECO2.2000.2021.csv"
df = pd.read_csv(inputfile)
sig = df.ultimoPrecio

power, freqs= matplotlib.mlab.psd(sig, NFFT)
plt.plot(power[10:]);plt.show()
plt.plot(freqs[10:],power[10:]);plt.show()


NFFT2=2500
freqs=np.arange(NFFT/2+1)*7/NFFT
f,x=plt.psd(sig, Fs=7, NFFT=NFFT2)
plt.plot(freqs,f);plt.show()

