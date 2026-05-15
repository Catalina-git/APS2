import numpy as np
import matplotlib.pyplot as plt

from scipy import signal as sig

import scipy.io as sio
from scipy.io.wavfile import write

# Crece la energia --> crece la varianza
# La flattop no es una ventana util para hacer estimacion espectral, porque tiene el lobulo principal muy ancho 
# PERIODOGRAM
# detrend = Flase --> xq es se;al sin ruido 
# scalling = 'density' --> por default, da en V cuadrado = watts

# WELCH
# average = 'mean'

#%% ECG SIN RUIDO

ecg_one_lead = np.load('ecg_sin_ruido.npy')
N_ECG = len(ecg_one_lead)
nn_ECG = np.arange(N_ECG)

plt.figure()
plt.plot(nn_ECG, ecg_one_lead)
plt.title("ECG sin ruido")

fs_ecg = 1000 # Hz
cant_promedio = 10
nperseg = ecg_one_lead.shape[0] // cant_promedio

f, Pxx = sig.welch(ecg_one_lead, fs = fs_ecg, window ='hamming', nperseg = nperseg, noverlap = 5)

plt.figure()
plt.title("WELCH - ECG")
plt.plot(f, Pxx)
plt.xlim(-1, 45)

#%% PPG SIN RUIDO

ppg = np.load('ppg_sin_ruido.npy')

plt.figure()
plt.plot(ppg)

fs_ppg = 400 # Hz

f_ppg, Pxx_ppg = sig.welch(ppg, fs = fs_ppg, window ='hamming', nperseg = nperseg, noverlap = 5)

plt.figure()
plt.title("WELCH - PPG")
plt.plot(f_ppg, Pxx_ppg)
plt.xlim(-1, 45)

#%% AUDIO 
# LA CUCARACHA
fs_audio1, wav_data1 = sio.wavfile.read('la cucaracha.wav')

plt.figure()
plt.title("Sonido de la cucaracha")
plt.plot(wav_data1)

cant_promedio_sonido = 100 
nperseg = wav_data1.shape[0] // cant_promedio_sonido

f_audio1, Pxx_audio1 = sig.welch(wav_data1, fs = fs_audio1, window ='hamming', nperseg = nperseg)

plt.figure()
plt.title("WELCH - AUDIO")
plt.plot(f_audio1, Pxx_audio1)
plt.xlim(-1, 2500)
