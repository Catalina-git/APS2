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
fs_ecg = 1000 # Hz
N_ECG = len(ecg_one_lead)
nn_ECG = np.arange(N_ECG)

plt.figure()
plt.plot(nn_ECG, ecg_one_lead)
plt.title("ECG sin ruido")
plt.show()

# WELCH
cant_promedios_ecg = 10
nperseg = N_ECG // cant_promedios_ecg

f_ecg, Pxx_ecg = sig.welch(ecg_one_lead, fs = fs_ecg, window ='hann', nperseg = nperseg, noverlap = 10)

plt.figure()
plt.title("WELCH - ECG")
plt.plot(f_ecg, Pxx_ecg)
plt.xlabel('Frecuencia [Hz]')
plt.xlim(-1, 45)
plt.grid()
plt.show()

# # Ahora hago welch pero con zero padding para suavizar el espectro 
# nfft = 10 * nperseg

# f_ecg_zp, Pxx_ecg_zp = sig.welch(ecg_one_lead, fs = fs_ecg, window ='hann', nperseg = nperseg, nfft = nfft)

# plt.figure()
# plt.title("WELCH CON ZERO PADDING - ECG")
# plt.plot(f_ecg_zp, Pxx_ecg_zp)
# plt.xlabel('Frecuencia [Hz]')
# plt.xlim(-1, 45)
# plt.grid()
# plt.show()

# Calculo el ancho de banda 
# Para pasa-banda: dos cortes
df_ecg = f_ecg[1] - f_ecg[0]
acum = np.cumsum(Pxx_ecg) * df_ecg
acum_norm = acum / acum[-1]

idx_inf_ecg = np.where(acum_norm >= 0.005)[0][0] # Límite inferior (0.5%)
idx_sup_ecg = np.where(acum_norm >= 0.995)[0][0] # Límite superior (99.5%)

f_inf_ecg = f_ecg[idx_inf_ecg]
f_sup_ecg = f_ecg[idx_sup_ecg]
BW_ecg = f_sup_ecg - f_inf_ecg

print(f"Ancho de bandadel ECG: BW = {BW_ecg:.1f} Hz")

#%% PPG SIN RUIDO
# Pasa bajos 

ppg = np.load('ppg_sin_ruido.npy')
fs_ppg = 400 # Hz
N_PPG = len(ppg)

plt.figure()
plt.title("PPG")
plt.plot(ppg)

cant_promedios_ppg = 10
nperseg_ppg = N_PPG // cant_promedios_ppg

f_ppg, Pxx_ppg = sig.welch(ppg, fs = fs_ppg, window ='hann', nperseg = nperseg_ppg)

# # Zero padding 
# nfft_ppg = 10 * nperseg_ppg
# f_ppg_zp, Pxx_ppg_zp = sig.welch(ppg, fs = fs_ppg, window ='hann', nperseg = nperseg_ppg, nfft = nfft_ppg)

# plt.figure()
# plt.title("WELCH CON ZERO PADDING - PPG")
# plt.plot(f_ppg_zp, Pxx_ppg_zp)
# plt.xlim(-1, 45)
# plt.xlabel('Frecuencia [Hz]')
# plt.grid()
# plt.show()

# Calculo el ancho de banda 
# Para pasa-banda: dos cortes
df_ppg = f_ppg[1] - f_ppg[0]
acum = np.cumsum(Pxx_ppg) * df_ppg
acum_norm = acum / acum[-1]

idx_inf_ppg = np.where(acum_norm >= 0.005)[0][0] # Límite inferior (0.5%)
idx_sup_ppg = np.where(acum_norm >= 0.995)[0][0] # Límite superior (99.5%)

f_inf_ppg = f_ppg[idx_inf_ppg]
f_sup_ppg = f_ppg[idx_sup_ppg]
BW_ppg = f_sup_ppg - f_inf_ppg

print(f"Ancho de banda del PPG: BW = {BW_ppg:.1f} Hz")

#%% AUDIO 
# Pasa banda

# LA CUCARACHA
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
N_audio = len(wav_data)

plt.figure()
plt.title("Audio de la cucaracha")
plt.plot(wav_data)

cant_promedio_audio = 10 
nperseg_audio = N_audio // cant_promedio_audio

f_audio, Pxx_audio = sig.welch(wav_data, fs = fs_audio, window ='hamming', nperseg = nperseg_audio)

# # Zero padding 
# nfft_audio = 4 * nperseg_audio
# f_audio_zp, Pxx_audio_zp = sig.welch(wav_data, fs = fs_audio, window ='hamming', nperseg = nperseg_audio, nfft = nfft_audio)

# plt.figure()
# plt.title("WELCH CON ZERO PADDING - AUDIO")
# plt.plot(f_audio_zp, Pxx_audio_zp)
# plt.xlim(-1, 2500)
# plt.xlabel('Frecuencia [Hz]')
# plt.grid()
# plt.show()

# Calculo el ancho de banda 
# Pasa-banda -> DOS cortes
df_audio = f_audio[1] - f_audio[0]
acum = np.cumsum(Pxx_audio) * df_audio
acum_norm = acum / acum[-1]

idx_inf_audio = np.where(acum_norm >= 0.005)[0][0]   # limite inferior (0.5%)
idx_sup_audio = np.where(acum_norm >= 0.995)[0][0]   # limite superior (99.5%)

f_inf_audio = f_audio[idx_inf_audio]
f_sup_audio = f_audio[idx_sup_audio]
BW_audio = f_sup_audio - f_inf_audio

print(f"Ancho de banda del audio de la cucaracha:  BW = {BW_audio:.1f} Hz")


#%% Para estimar el ancho de banda
# Hago la potencia acumulada a izquierda, y cuando llego al final del ancho de banda tengo que haber acumulado el total de la potencia
# Acumulo el 95% de la potencia (ahi tengo el ancho de banda) --> de toda mi potencia, 5% que queda es ruido 
# Pero si la se;al no tiene ruido? En la practica decir que no tiene ruido es decir que no tiene ruido considerable 
# Por eso hablo del 5%, 3%, 2% de la potencia 
# Entonces hablo de un contenido espectral en cuanto a la potencia bajisimo 

# %% Tabla con los resultados del ancho de banda
data = [
    ["ECG", f"{BW_ecg:.2f} Hz"],
    ["PPG", f"{BW_ppg:.2f} Hz"],
    ["AUDIO (La cucaracha)", f"{BW_audio:.2f} Hz"],
]

fig, ax = plt.subplots(figsize = (20, 1.5))
ax.axis('off')

tabla = ax.table(cellText = data, colLabels = ["Señal", "Ancho de banda"], cellLoc = 'center', loc = 'center')

# Esto es para poner en negrita los "titulos"
tabla[0,0].get_text().set_fontweight('bold')
tabla[0,1].get_text().set_fontweight('bold')

tabla.scale(1, 1.5)
tabla.auto_set_font_size(False)
tabla.set_fontsize(12)

plt.title("RESULTADOS", fontweight = 'bold', pad=10)
plt.show()

#%% Hago el mismo analisis pero con Blackman - Tukey

def blackman_tukey(x, fs, M=None):
    N = len(x)
    if M is None:
        M = N // 20
    M = min(M, N // 2 - 1)
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)                       # quito la continua

    r_full = np.correlate(x, x, mode='full') / N
    mid = len(r_full) // 2
    r = r_full[mid - (M - 1): mid + M]       # autocorrelacion simetrica, largo 2M-1

    window = sig.windows.blackman(len(r))
    r_windowed = r * window

    # CLAVE: reordeno para que el lag 0 quede en el indice 0 (fftshift),
    # asi la FFT no introduce fase espuria.
    Px = np.abs(np.fft.fft(np.fft.ifftshift(r_windowed), n=N))

    f = np.fft.fftfreq(N, d=1/fs)[:N // 2]
    Px = Px[:N // 2]
    return f, Px

# ECG
f_ecg_bt, Px_ecg_bt = blackman_tukey(ecg_one_lead, fs_ecg, M = N_ECG // 20)
 
plt.figure()
plt.title("Blackman-Tukey - ECG")
plt.plot(f_ecg_bt, Px_ecg_bt)
plt.xlim(-1, 45)
plt.xlabel('Frecuencia [Hz]')
plt.grid()
plt.show()

# PPG
f_ppg_bt, Px_ppg_bt = blackman_tukey(ppg, fs_ppg, M=N_PPG // 20)

plt.figure()
plt.title("Blackman-Tukey - PPG")
plt.plot(f_ppg_bt, Px_ppg_bt, label='Blackman-Tukey', alpha=0.8)
plt.xlim(-1, 45) 
plt.xlabel('Frecuencia [Hz]')
plt.legend() 
plt.grid()
plt.show()

# AUDIO 
f_audio_bt, Px_audio_bt = blackman_tukey(wav_data, fs_audio, M=N_audio // 100)

plt.figure()
plt.title("Blackman-Tukey - AUDIO")
plt.plot(f_audio_bt, Px_audio_bt, label='Blackman-Tukey', alpha=0.8)
plt.xlim(-1, 2500)
plt.xlabel('Frecuencia [Hz]')
plt.legend()
plt.grid()
plt.show()
 
#%% ANALISIS DE SESGO Y VARIANZA (sin realizaciones)
# Varianza teorica del estimador de Welch:  Var[P(f)] ~ P(f)^2 / K
#   K = cantidad de segmentos promediados (cant_promedio)
#   -> a mayor K, menor varianza (estimador mas suave)
# El sesgo se ve cualitativamente como el ensanchamiento de los picos
# al achicar los segmentos (peor resolucion espectral).
 

print("\nANALISIS DE SESGO Y VARIANZA\n")
 
var_ecg_W = (Pxx_ecg ** 2 / cant_promedios_ecg).mean()
var_ppg_W = (Pxx_ppg ** 2 / cant_promedios_ppg).mean()
var_aud_W = (Pxx_audio ** 2 / cant_promedio_audio).mean()

# Blackman-Tukey (uso el mismo M que pasaste a la funcion)
var_ecg_bt   = var_blackman_tukey(Px_ecg_bt,   N_ECG,   N_ECG // 20)
var_ppg_bt   = var_blackman_tukey(Px_ppg_bt,   N_PPG,   N_PPG // 20)
var_audio_bt = var_blackman_tukey(Px_audio_bt, N_audio, N_audio // 100)
 
print("\nVARIANZA DEL ESTIMADOR: Welch vs Blackman-Tukey\n")
print(f"{'Senal':<12}{'Var Welch':<18}{'Var Blackman-Tukey':<20}")
print(f"{'ECG':<12}{var_ecg_W:<18.3e}{var_ecg_bt:<20.3e}")
print(f"{'PPG':<12}{var_ppg_W:<18.3e}{var_ppg_bt:<20.3e}")
print(f"{'Audio':<12}{var_audio_W:<18.3e}{var_audio_bt:<20.3e}")
