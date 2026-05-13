import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

# Parámetros
N = 1000   # cantidad de muestras
R = 200    # cantidad de realizaciones
fs = N     # frecuencia de muestreo
df = 2*np.pi / N
Ts = 1 / fs
a0 = np.sqrt(2)    # amplitud
omega0 = N/4

SNR = 10 # en dB
Ps = a0**2 / 2
Pr = Ps / (10**(SNR/10))
desvio = np.sqrt(Pr)

na = np.random.normal(0, desvio, size = (R, N)) # ruido

fr = np.random.uniform(-2,2, size = R)

# Eje de muestras
n = np.arange(N)
n = n.reshape(1, N)

t = np.arange(N) / fs
tt = np.tile(t, (R,1))   # matriz 200x1000

omega1 = omega0 + fr * (2*np.pi/N)
omega1 = omega1.reshape(R, 1)

ff = np.arange(N) # Vector en frecuencia al escalar las muestras por la resolucion espectral
f = omega1*n
ff = ff.reshape(1000, 1)

# Matriz de 200 x 1000 (R x N)
matriz = f * tt

# Señal
s_mat = a0 * np.sin(2*np.pi*f*Ts) + na

# Traspongo la señal para poder graficarla 
señal_t = np.transpose(s_mat)

plt.figure()
plt.title("Senoidal + ruido (SNR = 10dB)")
plt.plot(t, señal_t)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.grid()
plt.show()

# FFT
eps = 1e-12
S_mat = np.fft.fft(s_mat, axis = 1) / N
S_mat_modulo = np.abs(S_mat)
S_mat_dB = 10 * np.log10(2*(S_mat_modulo)**2 + eps)
S_mat_dB = np.transpose(S_mat_dB)

plt.figure()
plt.title("FFT senoidal + ruido (SNR = 10dB)")
plt.plot(ff, S_mat_dB)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.xlim(0, fs/2)
plt.grid()
plt.show()

#%% SNR = 3dB
SNR = 3 # en dB
Ps = a0**2 / 2
Pr = Ps / (10**(SNR/10))
desvio = np.sqrt(Pr)

na = np.random.normal(0, desvio, size = (R, N)) # ruido

# Señal
s_mat1 = a0 * np.sin(2*np.pi*f*Ts) + na 

# Traspongo la señal para poder graficarla 
señal_t1 = np.transpose(s_mat1)

plt.figure()
plt.title("Senoidal + ruido (SNR = 3dB)")
plt.plot(t, señal_t1)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.grid()
plt.show()

# FFT
eps = 1e-12
S_mat1 = np.fft.fft(s_mat1, axis = 1) / N
S_mat_modulo1 = np.abs(S_mat1)
S_mat_dB1 = 10 * np.log10(2*(S_mat_modulo1)**2 + eps)
S_mat_dB1 = np.transpose(S_mat_dB1)

plt.figure()
plt.title("FFT senoidal + ruido (SNR = 3dB)")
plt.plot(ff, S_mat_dB1)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.xlim(0, fs/2)
plt.grid()
plt.show()

#%% Ventaneo SNR = 10dB 
SNR = 10
Ps = a0**2 / 2
Pr = Ps / (10**(SNR/10))
desvio = np.sqrt(Pr)

na = np.random.normal(0, desvio, size=(R, N))

s_mat_10 = a0 * np.sin(2 * np.pi * f * Ts) + na  # (R, N)

eps = 1e-12

# RECTANGULAR
s_vent_R10 = s_mat_10 * windows.boxcar(N)

S_vent_R10 = np.fft.fft(s_vent_R10, axis=1) / N
S_vent_modulo_R10 = np.abs(S_vent_R10)
S_vent_R10_dB = 10 * np.log10(2*(S_vent_modulo_R10)**2 + eps)
S_vent_R10_dB = np.transpose(S_vent_R10_dB)

# FLATTOP
s_vent_F10 = s_mat_10 * windows.flattop(N)

S_vent_F10 = np.fft.fft(s_vent_F10, axis=1) / N
S_vent_modulo_F10 = np.abs(S_vent_F10)
S_vent_F10_dB = 10 * np.log10(2*(S_vent_modulo_F10)**2 + eps)
S_vent_F10_dB = np.transpose(S_vent_F10_dB)

# BLACKMANHARRIS
s_vent_B10 = s_mat_10 * windows.blackmanharris(N)

S_vent_B10 = np.fft.fft(s_vent_B10, axis=1) / N
S_vent_modulo_B10 = np.abs(S_vent_B10)
S_vent_B10_dB = 10 * np.log10(2*(S_vent_modulo_B10)**2 + eps)
S_vent_B10_dB = np.transpose(S_vent_B10_dB)

# HAMMING
s_vent_H10 = s_mat_10 * windows.hamming(N)

S_vent_H10 = np.fft.fft(s_vent_H10, axis=1) / N
S_vent_modulo_H10 = np.abs(S_vent_H10)
S_vent_H10_dB = 10 * np.log10(2*(S_vent_modulo_H10)**2 + eps)
S_vent_H10_dB = np.transpose(S_vent_H10_dB)

# Graficos
plt.figure()
plt.suptitle("Señal con ruido 10dB ventaneada")

plt.subplot(2,2,1)
plt.title("RECTANGULAR")
plt.plot(ff, S_vent_R10_dB)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.xlim(0, fs/2)
plt.grid()

plt.subplot(2,2,2)
plt.title("FLATTOP")
plt.plot(ff, S_vent_F10_dB)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.xlim(0, fs/2)
plt.grid()

plt.subplot(2,2,3)
plt.title("BLACKMANHARRIS")
plt.plot(ff, S_vent_B10_dB)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.xlim(0, fs/2)
plt.grid()

plt.subplot(2,2,4)
plt.title("HAMMING")
plt.plot(ff, S_vent_H10_dB)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.xlim(0, fs/2)
plt.grid()

plt.tight_layout()
plt.show()

#%% Ventaneo SNR = 3dB
SNR = 3
Ps = a0**2 / 2
Pr = Ps / (10**(SNR/10))
desvio = np.sqrt(Pr)

na = np.random.normal(0, desvio, size=(R, N))

s_mat_3 = a0 * np.sin(2 * np.pi * f * Ts) + na  # (R, N)

eps = 1e-12

# RECTANGULAR
s_vent_R3 = s_mat_3 * windows.boxcar(N)

S_vent_R3 = np.fft.fft(s_vent_R3, axis=1) / N
S_vent_modulo_R3 = np.abs(S_vent_R3)
S_vent_R3_dB = 10 * np.log10(2*(S_vent_modulo_R3)**2 + eps)
S_vent_R3_dB = np.transpose(S_vent_R3_dB)

# FLATTOP
s_vent_F3 = s_mat_3 * windows.flattop(N)

S_vent_F3 = np.fft.fft(s_vent_F3, axis=1) / N
S_vent_modulo_F3 = np.abs(S_vent_F3)
S_vent_F3_dB = 10 * np.log10(2*(S_vent_modulo_F3)**2 + eps)
S_vent_F3_dB = np.transpose(S_vent_F3_dB)

# BLACKMANHARRIS
s_vent_B3 = s_mat_3 * windows.blackmanharris(N)

S_vent_B3 = np.fft.fft(s_vent_B3, axis=1) / N
S_vent_modulo_B3 = np.abs(S_vent_B3)
S_vent_B3_dB = 10 * np.log10(2*(S_vent_modulo_B3)**2 + eps)
S_vent_B3_dB = np.transpose(S_vent_B3_dB)

# HAMMING
s_vent_H3 = s_mat_3 * windows.hamming(N)

S_vent_H3 = np.fft.fft(s_vent_H3, axis=1) / N
S_vent_modulo_H3 = np.abs(S_vent_H3)
S_vent_H3_dB = 10 * np.log10(2*(S_vent_modulo_H3)**2 + eps)
S_vent_H3_dB = np.transpose(S_vent_H3_dB)

# Graficos
plt.figure()
plt.suptitle("Señal con ruido 3dB ventaneada")

plt.subplot(2,2,1)
plt.title("RECTANGULAR")
plt.plot(ff, S_vent_R3_dB)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.xlim(0, fs/2)
plt.grid()

plt.subplot(2,2,2)
plt.title("FLATTOP")
plt.plot(ff, S_vent_F3_dB)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.xlim(0, fs/2)
plt.grid()

plt.subplot(2,2,3)
plt.title("BLACKMANHARRIS")
plt.plot(ff, S_vent_B3_dB)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.xlim(0, fs/2)
plt.grid()

plt.subplot(2,2,4)
plt.title("HAMMING")
plt.plot(ff, S_vent_H3_dB)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.xlim(0, fs/2)
plt.grid()

plt.tight_layout()
plt.show()

#%% Observaciones 
# Toda ventana tiene un compromiso (trade - off) entre dos caracteristicas: 
# - Resolucion espectral --> que tan angosto es el lobulo principal (pico)
# - Nivel de lobulos secundarios --> cuanto "se desparrama" la energia a frecuencias vecinas (leakage)
  
#%% Estimador de amplitud 
trans = 0.35
bins = 10

# Ganancias coherentes de cada ventana
cg_R = np.mean(windows.boxcar(N))
cg_F = np.mean(windows.flattop(N))
cg_B = np.mean(windows.blackmanharris(N))
cg_H = np.mean(windows.hamming(N))

# SNR = 10dB
estimador_a_R10 = 10*np.log10(2*(np.abs(S_vent_R10[:, N//4])/cg_R)**2)
estimador_a_F10 = 10*np.log10(2*(np.abs(S_vent_F10[:, N//4])/cg_F)**2)
estimador_a_B10 = 10*np.log10(2*(np.abs(S_vent_B10[:, N//4])/cg_B)**2)
estimador_a_H10 = 10*np.log10(2*(np.abs(S_vent_H10[:, N//4])/cg_H)**2)

# SNR = 3dB
estimador_a_R3 = 10*np.log10(2*(np.abs(S_vent_R3[:, N//4])/cg_R)**2)
estimador_a_F3 = 10*np.log10(2*(np.abs(S_vent_F3[:, N//4])/cg_F)**2)
estimador_a_B3 = 10*np.log10(2*(np.abs(S_vent_B3[:, N//4])/cg_B)**2)
estimador_a_H3 = 10*np.log10(2*(np.abs(S_vent_H3[:, N//4])/cg_H)**2)

# Histograma

plt.figure()
plt.suptitle("Histograma de la estimación de energía")

plt.subplot(1,2,1)
plt.title("SNR = 10dB")
plt.hist(estimador_a_R10, label = 'Rectangular', alpha = trans, bins = bins)
plt.hist(estimador_a_F10,label = 'Flattop', alpha = trans, bins = bins)
plt.hist(estimador_a_B10,label = 'Blackmanharris', alpha = trans, bins = bins)
plt.hist(estimador_a_H10,label = 'Hamming', alpha = trans, bins = bins)
plt.xlabel('PDS [db]')
plt.ylabel('#Cantidad de ocurrencias')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.title("SNR = 3dB")
plt.hist(estimador_a_R3, label = 'Rectangular', alpha = trans, bins = bins)
plt.hist(estimador_a_F3,label = 'Flattop', alpha = trans, bins = bins)
plt.hist(estimador_a_B3,label = 'Blackmanharris', alpha = trans, bins = bins)
plt.hist(estimador_a_H3,label = 'Hamming', alpha = trans, bins = bins)
plt.xlabel('PDS [db]')
plt.ylabel('#Cantidad de ocurrencias')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

#%% Estimador de la frecuencia 
# SNR = 10dB
estimador_f_R10 = np.argmax(np.abs(S_vent_R10[:, :N//2]), axis=1) * (fs/N)
estimador_f_F10 = np.argmax(np.abs(S_vent_F10[:, :N//2]), axis=1) * (fs/N)
estimador_f_B10 = np.argmax(np.abs(S_vent_B10[:, :N//2]), axis=1) * (fs/N)
estimador_f_H10 = np.argmax(np.abs(S_vent_H10[:, :N//2]), axis=1) * (fs/N)

# SNR = 3dB
estimador_f_R3 = np.argmax(np.abs(S_vent_R3[:, :N//2]), axis=1) * (fs/N)
estimador_f_F3 = np.argmax(np.abs(S_vent_F3[:, :N//2]), axis=1) * (fs/N)
estimador_f_B3 = np.argmax(np.abs(S_vent_B3[:, :N//2]), axis=1) * (fs/N)
estimador_f_H3 = np.argmax(np.abs(S_vent_H3[:, :N//2]), axis=1) * (fs/N)

# Histograma
plt.figure()
plt.suptitle("Histograma del estimador de frecuencia")

plt.subplot(1,2,1)
plt.title("SNR = 10dB") 
plt.hist(estimador_f_R10, label = 'Rectangular',    alpha = trans, bins = 30)
plt.hist(estimador_f_F10, label = 'Flattop',        alpha = trans, bins = 30)
plt.hist(estimador_f_B10, label = 'Blackmanharris', alpha = trans, bins = 30)
plt.hist(estimador_f_H10, label = 'Hamming',        alpha = trans, bins = 30)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('# Cantidad de ocurrencias')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.title("SNR = 3dB")
plt.hist(estimador_f_R3, label = 'Rectangular',    alpha = trans, bins = 30)
plt.hist(estimador_f_F3, label = 'Flattop',        alpha = trans, bins = 30)
plt.hist(estimador_f_B3, label = 'Blackmanharris', alpha = trans, bins = 30)
plt.hist(estimador_f_H3, label = 'Hamming',        alpha = trans, bins = 30)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('# Cantidad de ocurrencias')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

