import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write


#%%

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
# mat_struct = sio.loadmat('./ECG_TP4.mat')

# ecg_one_lead = mat_struct['ecg_lead']
# N = len(ecg_one_lead)

# hb_1 = mat_struct['heartbeat_pattern1']
# hb_2 = mat_struct['heartbeat_pattern2']

# plt.figure()
# plt.plot(ecg_one_lead[5000:12000])

# plt.figure()
# plt.plot(hb_1)

# plt.figure()
# plt.plot(hb_2)

##################
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')
N_ECG = len(ecg_one_lead)
nn_ECG = np.arange(N_ECG)

plt.figure()
plt.plot(nn_ECG, ecg_one_lead)
plt.title("ECG sin ruido")


cant_promedio = 10 # Si aumento la cantidad de promedio se ve mejor, mas suave, es lo subyacente 
# --> me saco de encima todo lo inorrelado, no es lo real 
# --> relacionado con la varianza 
# --> el promedio pongo lo que sea necesario para ver lo incorrelado
# Yo quiero menos varianza, pero tampoco quiero que mi centro de masa se corre
# Cuando se empieza a acabar la varianza es donde quiero verlo
nperseg = ecg_one_lead.shape[0] // cant_promedio

f, Pxx = sig.welch(ecg_one_lead, fs = fs_ecg, window ='hann', nperseg = nperseg)

plt.figure()
plt.plot(f, Pxx)
plt.xlim(0, 50)
plt.title("PDS del ECG sin ruido") # Este es el espectro en veces

# Le meto el zero padding para ver la suavidad
nfft = 10 * nperseg # Es la FFT con zero padding (con relleno de ceros)
f_padding, Pxx_padding = sig.welch(ecg_one_lead, fs = fs_ecg, window ='flattop', nperseg = nperseg, nfft = nfft)

df_ECG = fs_ecg / N_ECG
ff_ECG = np.arange(N_ECG) * df_ECG
df = f_padding[1] - f_padding[0]

energia_acum = np.cumsum(Pxx) * df # Esto me devuelve un vector de sumas acumuladas
energia_acum_normal = energia_acum / energia_acum[-1]
indice_corte = np.where (energia_acum_normal >= 0.99)[0][0] # Con [0][0] me devuelve el primer valor que cumple 
frecuencia_corte = ff_ECG[indice_corte]

print(f"Frecuencia de corte = {frecuencia_corte} Hz")

plt.figure()
plt.plot(f_padding, Pxx_padding)
plt.axvline(frecuencia_corte, linestyle='--', color='orange', label = f'Frecuencia de corte = {frecuencia_corte} Hz')
plt.title("PDS del ECG sin ruido con zero padding")
plt.legend()


#%%

####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

##################
## PPG con ruido
##################

# # Cargar el archivo CSV como un array de NumPy
# ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe


##################
## PPG sin ruido
##################

ppg = np.load('ppg_sin_ruido.npy')

plt.figure()
plt.plot(ppg)


#%%

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
# LA CUCARACHA
fs_audio1, wav_data1 = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')

plt.figure()
plt.title("Sonido de la cucaracha")
plt.plot(wav_data1)

# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
# sd.play(wav_data, fs_audio)

cant_promedio_sonido = 100 
nperseg = wav_data1.shape[0] // cant_promedio_sonido
nfft = 4 * nperseg

f_welch1, pds_welch1 = sig.welch(x = wav_data1, fs = fs_audio1, window = "hamming", nperseg = nperseg)

plt.figure()
plt.title("PDS del sonido")
plt.xlim(0, 5000)
plt.plot(f_welch1, pds_welch1)

# Aplico zero padding
f_welch1_padding, pds_welch1_padding = sig.welch(x = wav_data1, fs = fs_audio1, window = "hamming", nperseg = nperseg, nfft = nfft)

plt.figure()
plt.title("PDS del sonido con zero padding")
plt.xlim(0, 5000)
plt.plot(f_welch1_padding, pds_welch1_padding)


# SILBIDO
fs_audio3, wav_data3 = sio.wavfile.read('silbido.wav')

plt.figure()
plt.title("Sonido del silbido")
plt.plot(wav_data3)

f_welch3, pds_welch3 = sig.welch(x = wav_data3, fs = fs_audio3, window = "hamming", nperseg = nperseg)

plt.figure()
plt.title("PDS del silbido")
plt.xlim(0, 5000)
plt.plot(f_welch3, pds_welch3)

# Aplico zero padding
f_welch3_padding, pds_welch3_padding = sig.welch(x = wav_data3, fs = fs_audio3, window = "hamming", nperseg = nperseg, nfft = nfft)

plt.figure()
plt.title("PDS del silbido con zero padding")
plt.xlim(0, 5000)
plt.plot(f_welch3_padding, pds_welch3_padding)

# %% TS5 --> anho de banda de la cucaracha
# La cucaracha (audio)
# df_audio = f_audio[1] - f_audio[0]
# energia_acum_audio = np.cumsum(Px_audio) * df_audio
# energia_acum_audio_norm = energia_acum_audio / energia_acum_audio[-1]
# indice_corte_audio = np.where(energia_acum_audio_norm >= 0.99)[0][0]
# frecuencia_corte_audio = f_audio[indice_corte_audio]

# plt.figure()
# plt.plot(f_audio, Px_audio, label = 'PSD la cucaracha')
# plt.axvline(frecuencia_corte_audio, color='orange', linestyle='--', label=f'Fc = {frecuencia_corte_audio:.2f} Hz')
# plt.title("PSD Audio + Frecuencia de corte (99%)")
# plt.xlabel("Frecuencia [Hz]")
# plt.ylabel("Densidad espectral de potencia")
# plt.legend()
# plt.grid(True)
# plt.show()
