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
# Pasa bajos

ecg_one_lead = np.load('ecg_sin_ruido.npy')
N_ECG = len(ecg_one_lead)
nn_ECG = np.arange(N_ECG)

# N = 1000

plt.figure()
plt.plot(nn_ECG, ecg_one_lead)
plt.title("ECG sin ruido")
plt.show()

fs_ecg = 1000 # Hz
cant_promedio = 10
nperseg = ecg_one_lead.shape[0] // cant_promedio

f, Pxx = sig.welch(ecg_one_lead, fs = fs_ecg, window ='hamming', nperseg = nperseg, noverlap = 10)

plt.figure()
plt.title("WELCH - ECG")
# plt.plot(f, Pxx)
plt.xlim(-1, 45)


cant_promedio = 30
nperseg = ecg_one_lead.shape[0] // cant_promedio

f, Pxx = sig.welch(ecg_one_lead, fs = fs_ecg, window ='hamming', nperseg = nperseg, noverlap = 15)

plt.plot(f, Pxx)

plt.show()

#%% PPG SIN RUIDO
# Pasa bajos 

ppg = np.load('ppg_sin_ruido.npy')

plt.figure()
plt.title("PPG")
plt.plot(ppg)

fs_ppg = 400 # Hz

f_ppg, Pxx_ppg = sig.welch(ppg, fs = fs_ppg, window ='hamming', nperseg = nperseg, noverlap = 5)

plt.figure()
plt.title("WELCH - PPG")
plt.plot(f_ppg, Pxx_ppg)
plt.xlim(-1, 45)

#%% AUDIO 
# Pasa banda

# LA CUCARACHA
fs_audio1, wav_data1 = sio.wavfile.read('la cucaracha.wav')

plt.figure()
plt.title("Audio de la cucaracha")
plt.plot(wav_data1)

cant_promedio_sonido = 100 
nperseg = wav_data1.shape[0] // cant_promedio_sonido

f_audio1, Pxx_audio1 = sig.welch(wav_data1, fs = fs_audio1, window ='hamming', nperseg = nperseg, noverlap = 50)

plt.figure()
plt.title("WELCH - AUDIO")
plt.plot(f_audio1, Pxx_audio1)
plt.xlim(-1, 2500)

#%% Para estimar el ancho de banda
# Hago la potencia acumulada a izquierda, y cuando llego al final del ancho de banda tengo que haber acumulado el total de la potencia
# Acumulo el 95% de la potencia (ahi tengo el ancho de banda) --> de toda mi potencia, 5% que queda es ruido 
# Pero si la se;al no tiene ruido? En la practica decir que no tiene ruido es decir que no tiene ruido considerable 
# Por eso hablo del 5%, 3%, 2% de la potencia 
# Entonces hablo de un contenido espectral en cuanto a la potencia bajisimo 

