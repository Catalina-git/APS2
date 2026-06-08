import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

# Parámetros
N = 1000 # cantidad de muestras
R = 200 # cantidad de realizaciones
fs = 1000 # frecuencia de muestreo
df = fs / N
Ts = 1 / fs
a0 = np.sqrt(2) # amplitud
omega0 = N/4
fr = np.random.uniform(-2,2, size = R)

# Eje de muestras
n = np.arange(N)
n = n.reshape(1, N)

t = np.arange(N) / fs
tt = np.tile(t, (R,1))   # matriz 200x1000

omega1 = omega0 + fr * df
omega1 = omega1.reshape(R, 1)

ff = np.arange(N) # Vector en frecuencia al escalar las muestras por la resolucion espectral
f = omega1*n
ff = ff.reshape(1000, 1)


#%% SNR = 10dB
SNR = 10 # en dB
Ps = a0**2 / 2
Pr = Ps / (10**(SNR/10))
desvio = np.sqrt(Pr)

na = np.random.normal(0, desvio, size = (R, N)) # ruido

# Señal
s_mat_10 = a0 * np.sin(2 * np.pi * f * Ts) + na

# Traspongo la señal para poder graficarla 
señal_t10 = np.transpose(s_mat_10)

plt.figure()
plt.title("Senoidal + ruido (SNR = 10dB)")
plt.plot(t, señal_t10)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.grid()
plt.show()

# FFT
eps = 1e-12
S_mat_10 = np.fft.fft(s_mat_10, axis = 1) / N
S_mat_modulo10 = np.abs(S_mat_10)
S_mat_dB10 = 10 * np.log10(2*(S_mat_modulo10)**2 + eps)
S_mat_dB10 = np.transpose(S_mat_dB10)

plt.figure()
plt.title("FFT senoidal + ruido (SNR = 10dB)")
plt.plot(ff, S_mat_dB10)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.xlim(0, fs/2)
plt.grid()
plt.show()

#%% Ventaneo SNR = 10dB 
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

#%% Ganancia coherente de cada ventana
CG_R = np.mean(windows.boxcar(N))
CG_F = np.mean(windows.flattop(N))
CG_B = np.mean(windows.blackmanharris(N))
CG_H = np.mean(windows.hamming(N))

# Calculo el maximo espectral 
kmax_R10 = np.argmax(np.abs(S_vent_R10[:, :N//2]), axis=1)
kmax_F10 = np.argmax(np.abs(S_vent_F10[:, :N//2]), axis=1)
kmax_B10 = np.argmax(np.abs(S_vent_B10[:, :N//2]), axis=1)
kmax_H10 = np.argmax(np.abs(S_vent_H10[:, :N//2]), axis=1)

#%% Estimador de amplitud SNR = 10dB
# SNR = 10dB
estimador_a_R10 = 2*np.abs(S_vent_R10[np.arange(R), kmax_R10])/CG_R
estimador_a_F10 = 2*np.abs(S_vent_F10[np.arange(R), kmax_F10])/CG_F
estimador_a_B10 = 2*np.abs(S_vent_B10[np.arange(R), kmax_B10])/CG_B
estimador_a_H10 = 2*np.abs(S_vent_H10[np.arange(R), kmax_H10])/CG_H

#%% SESGO Y VARIANZA SNR = 10dB
sesgo_amp_R10 = np.mean(estimador_a_R10) - a0 
sesgo_amp_F10 = np.mean(estimador_a_F10) - a0 
sesgo_amp_B10 = np.mean(estimador_a_B10) - a0 
sesgo_amp_H10 = np.mean(estimador_a_H10) - a0 

varianza_amp_R10 = np.var(estimador_a_R10)
varianza_amp_F10 = np.var(estimador_a_F10)
varianza_amp_B10 = np.var(estimador_a_B10)
varianza_amp_H10 = np.var(estimador_a_H10)

#%% Estimador de la frecuencia SNR = 10dB
estimador_f_R10 = np.argmax(np.abs(S_vent_R10[:, :N//2]), axis = 1)
estimador_f_F10 = np.argmax(np.abs(S_vent_F10[:, :N//2]), axis = 1)
estimador_f_B10 = np.argmax(np.abs(S_vent_B10[:, :N//2]), axis = 1)
estimador_f_H10 = np.argmax(np.abs(S_vent_H10[:, :N//2]), axis = 1)

#%% SESGO Y VARIANZA SNR = 10dB
sesgo_frec_R10 = np.mean(estimador_f_R10) - np.mean(omega1)
sesgo_frec_F10 = np.mean(estimador_f_F10) - np.mean(omega1)
sesgo_frec_B10 = np.mean(estimador_f_B10) - np.mean(omega1)
sesgo_frec_H10 = np.mean(estimador_f_H10) - np.mean(omega1)

varianza_frec_R10 = np.var(estimador_f_R10)
varianza_frec_F10 = np.var(estimador_f_F10)
varianza_frec_B10 = np.var(estimador_f_B10)
varianza_frec_H10 = np.var(estimador_f_H10)

#%% Efecto del zero padding en el estimador de frecuencia (SNR = 10dB)
N_zp = 2 * N  # 2000 puntos

# Rectangular
zp_W_R10 = np.fft.fft(s_vent_R10, n = N_zp, axis = 1)
estimador_f_zp_R10 = np.argmax(np.abs(zp_W_R10[:, :N_zp//2]), axis = 1) / 2
# Dividir por 2 porque los bins ahora son la mitad de anchos

# Flattop
zp_W_F10 = np.fft.fft(s_vent_F10, n = N_zp, axis  =1)
estimador_f_zp_F10 = np.argmax(np.abs(zp_W_F10[:, :N_zp//2]), axis = 1) / 2

# Blackmanharris
zp_W_B10 = np.fft.fft(s_vent_B10, n = N_zp, axis = 1)
estimador_f_zp_B10 = np.argmax(np.abs(zp_W_B10[:, :N_zp//2]), axis = 1) / 2

# Hamming
zp_W_H10 = np.fft.fft(s_vent_H10, n = N_zp, axis = 1)
estimador_f_zp_H10 = np.argmax(np.abs(zp_W_H10[:, :N_zp//2]), axis = 1) / 2

# Sesgo y varianza CON zero padding
sesgo_f_zp_R10 = np.mean(estimador_f_zp_R10) - np.mean(omega1)
sesgo_f_zp_F10 = np.mean(estimador_f_zp_F10) - np.mean(omega1)
sesgo_f_zp_B10 = np.mean(estimador_f_zp_B10) - np.mean(omega1)
sesgo_f_zp_H10 = np.mean(estimador_f_zp_H10) - np.mean(omega1)

var_f_zp_R10 = np.var(estimador_f_zp_R10)
var_f_zp_F10 = np.var(estimador_f_zp_F10)
var_f_zp_B10 = np.var(estimador_f_zp_B10)
var_f_zp_H10 = np.var(estimador_f_zp_H10)

#%% Histograma comparativo ZP vs sin ZP
plt.figure(figsize=(12, 5))
plt.suptitle("Zero Padding: efecto en estimador de frecuencia (SNR = 10dB)")

plt.subplot(1, 2, 1)
plt.title("SIN zero padding")
plt.hist(estimador_f_R10, label = 'Rectangular', alpha = 0.4, bins = 15)
plt.hist(estimador_f_F10,  label = 'Flattop', alpha = 0.4, bins = 15)
plt.hist(estimador_f_B10,  label = 'Blackmanharris', alpha = 0.4, bins = 15)
plt.hist(estimador_f_H10,  label = 'Hamming', alpha = 0.4, bins = 15)
plt.axvline(np.mean(omega1), color = 'k', linestyle = '--', label = 'Ω1 verdadera')
plt.xlabel('Bin de frecuencia')
plt.ylabel('Ocurrencias')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.title("CON zero padding")
plt.hist(estimador_f_zp_R10,      label = 'Rectangular', alpha = 0.4, bins = 15)
plt.hist(estimador_f_zp_F10,     label='Flattop', alpha = 0.4, bins = 15)
plt.hist(estimador_f_zp_B10, label='Blackmanharris', alpha = 0.4, bins = 15)
plt.hist(estimador_f_zp_H10,      label='Hamming',        alpha = 0.4, bins = 15)
plt.axvline(np.mean(omega1), color = 'k', linestyle = '--', label = 'Ω1 verdadera')
plt.xlabel('Bin de frecuencia')
plt.ylabel('Ocurrencias')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

#%% SNR = 3dB
SNR = 3 # en dB
Ps = a0**2 / 2
Pr = Ps / (10**(SNR/10))
desvio = np.sqrt(Pr)

na = np.random.normal(0, desvio, size = (R, N)) # ruido

# Señal
s_mat_3 = a0 * np.sin(2 * np.pi * f * Ts) + na 

# Traspongo la señal para poder graficarla 
señal_t3 = np.transpose(s_mat_3)

plt.figure()
plt.title("Senoidal + ruido (SNR = 3dB)")
plt.plot(t, señal_t3)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.grid()
plt.show()

# FFT
eps = 1e-12
S_mat_3 = np.fft.fft(s_mat_3, axis = 1) / N
S_mat_modulo3 = np.abs(S_mat_3)
S_mat_dB3 = 10 * np.log10(2*(S_mat_modulo3)**2 + eps)
S_mat_dB3 = np.transpose(S_mat_dB3)

plt.figure()
plt.title("FFT senoidal + ruido (SNR = 3dB)")
plt.plot(ff, S_mat_dB3)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]') # Es la densidad espectral de potencia 
plt.xlim(0, fs/2)
plt.grid()
plt.show()

#%% Ventaneo SNR = 3dB
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

#%% Calculo el maximo espectral 
kmax_R3 = np.argmax(np.abs(S_vent_R3[:, :N//2]), axis=1)
kmax_F3 = np.argmax(np.abs(S_vent_F3[:, :N//2]), axis=1)
kmax_B3 = np.argmax(np.abs(S_vent_B3[:, :N//2]), axis=1)
kmax_H3 = np.argmax(np.abs(S_vent_H3[:, :N//2]), axis=1)

#%% Estimador de amplitud SNR = 3dB
estimador_a_R3 = 2*np.abs(S_vent_R3[np.arange(R), kmax_R3])/CG_R
estimador_a_F3 = 2*np.abs(S_vent_F3[np.arange(R), kmax_F3])/CG_F
estimador_a_B3 = 2*np.abs(S_vent_B3[np.arange(R), kmax_B3])/CG_B
estimador_a_H3 = 2*np.abs(S_vent_H3[np.arange(R), kmax_H3])/CG_H

#%% SESGO Y VARIANZA SNR = 3dB
sesgo_R3 = np.mean(estimador_a_R3) - a0 
sesgo_F3 = np.mean(estimador_a_F3) - a0 
sesgo_B3 = np.mean(estimador_a_B3) - a0 
sesgo_H3 = np.mean(estimador_a_H3) - a0 

varianza_R3 = np.var(estimador_a_R3)
varianza_F3 = np.var(estimador_a_F3)
varianza_B3 = np.var(estimador_a_B3)
varianza_H3 = np.var(estimador_a_H3)

#%% Estimador de la frecuencia SNR = 3dB
estimador_f_R3 = np.argmax(np.abs(S_vent_R3[:, :N//2]), axis=1)
estimador_f_F3 = np.argmax(np.abs(S_vent_F3[:, :N//2]), axis=1)
estimador_f_B3 = np.argmax(np.abs(S_vent_B3[:, :N//2]), axis=1)
estimador_f_H3 = np.argmax(np.abs(S_vent_H3[:, :N//2]), axis=1)

#%% SESGO Y VARIANZA SNR = 3dB
sesgo_frec_R3 = np.mean(estimador_f_R3) - np.mean(omega1)
sesgo_frec_F3 = np.mean(estimador_f_F3) - np.mean(omega1)
sesgo_frec_B3 = np.mean(estimador_f_B3) - np.mean(omega1)
sesgo_frec_H3 = np.mean(estimador_f_H3) - np.mean(omega1)

varianza_frec_R3 = np.var(estimador_f_R3)
varianza_frec_F3 = np.var(estimador_f_F3)
varianza_frec_B3 = np.var(estimador_f_B3)
varianza_frec_H3 = np.var(estimador_f_H3)

#%% Efecto del zero padding en el estimador de frecuencia (SNR = 3dB)
N_zp = 2 * N  # 2000 puntos

# Rectangular
zp_W_R3 = np.fft.fft(s_vent_R3, n = N_zp, axis = 1)
estimador_f_zp_R3 = np.argmax(np.abs(zp_W_R3[:, :N_zp//2]), axis = 1) / 2
# Dividir por 2 porque los bins ahora son la mitad de anchos

# Flattop
zp_W_F3 = np.fft.fft(s_vent_F3, n = N_zp, axis  =1)
estimador_f_zp_F3 = np.argmax(np.abs(zp_W_F3[:, :N_zp//2]), axis = 1) / 2

# Blackmanharris
zp_W_B3 = np.fft.fft(s_vent_B3, n = N_zp, axis = 1)
estimador_f_zp_B3 = np.argmax(np.abs(zp_W_B3[:, :N_zp//2]), axis = 1) / 2

# Hamming
zp_W_H3 = np.fft.fft(s_vent_H3, n = N_zp, axis = 1)
estimador_f_zp_H3 = np.argmax(np.abs(zp_W_H3[:, :N_zp//2]), axis = 1) / 2

# Sesgo y varianza CON zero padding
sesgo_f_zp_R3 = np.mean(estimador_f_zp_R3) - np.mean(omega1)
sesgo_f_zp_F3 = np.mean(estimador_f_zp_F3) - np.mean(omega1)
sesgo_f_zp_B3 = np.mean(estimador_f_zp_B3) - np.mean(omega1)
sesgo_f_zp_H3 = np.mean(estimador_f_zp_H3) - np.mean(omega1)

var_f_zp_R3 = np.var(estimador_f_zp_R3)
var_f_zp_F3 = np.var(estimador_f_zp_F3)
var_f_zp_B3 = np.var(estimador_f_zp_B3)
var_f_zp_H3 = np.var(estimador_f_zp_H3)

#%% Histograma comparativo ZP vs sin ZP
plt.figure(figsize=(12, 5))
plt.suptitle("Zero Padding: efecto en estimador de frecuencia (SNR = 3dB)")

plt.subplot(1, 2, 1)
plt.title("SIN zero padding")
plt.hist(estimador_f_R3, label = 'Rectangular', alpha = 0.4, bins = 15)
plt.hist(estimador_f_F3,  label = 'Flattop', alpha = 0.4, bins = 15)
plt.hist(estimador_f_B3,  label = 'Blackmanharris', alpha = 0.4, bins = 15)
plt.hist(estimador_f_H3,  label = 'Hamming', alpha = 0.4, bins = 15)
plt.axvline(np.mean(omega1), color = 'k', linestyle = '--', label = 'Ω1 verdadera')
plt.xlabel('Bin de frecuencia')
plt.ylabel('Ocurrencias')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.title("CON zero padding")
plt.hist(estimador_f_zp_R3, label = 'Rectangular', alpha = 0.4, bins = 15)
plt.hist(estimador_f_zp_F3, label='Flattop', alpha = 0.4, bins = 15)
plt.hist(estimador_f_zp_B3, label='Blackmanharris', alpha = 0.4, bins = 15)
plt.hist(estimador_f_zp_H3 , label='Hamming', alpha = 0.4, bins = 15)
plt.axvline(np.mean(omega1), color = 'k', linestyle = '--', label = 'Ω1 verdadera')
plt.xlabel('Bin de frecuencia')
plt.ylabel('Ocurrencias')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

#%% Observaciones 
# Toda ventana tiene un compromiso (trade - off) entre dos caracteristicas: 
# - Resolucion espectral --> que tan angosto es el lobulo principal (pico)
# - Nivel de lobulos secundarios --> cuanto "se desparrama" la energia a frecuencias vecinas (leakage)
# La flattop por definicion quita energia, la anchura en el histograma de estimadores de amplitud tiene que ser chica
  
#%% Histogramas de los estimadores de amplitud 
trans = 0.35
bins = 10

# Histograma

plt.figure()
plt.suptitle("Histograma de la estimación de energía")

plt.subplot(1,2,1)
plt.title("SNR = 10dB")
plt.hist(estimador_a_R10, label = 'Rectangular', alpha = trans, bins = bins)
plt.hist(estimador_a_F10,label = 'Flattop', alpha = trans, bins = bins)
plt.hist(estimador_a_B10,label = 'Blackmanharris', alpha = trans, bins = bins)
plt.hist(estimador_a_H10,label = 'Hamming', alpha = trans, bins = bins)
plt.axvline(a0, color = 'k', linestyle = '--', label = 'a\u2080 verdadera')
plt.ylabel('#Cantidad de ocurrencias')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.title("SNR = 3dB")
plt.hist(estimador_a_R3, label = 'Rectangular', alpha = trans, bins = bins)
plt.hist(estimador_a_F3,label = 'Flattop', alpha = trans, bins = bins)
plt.hist(estimador_a_B3,label = 'Blackmanharris', alpha = trans, bins = bins)
plt.hist(estimador_a_H3,label = 'Hamming', alpha = trans, bins = bins)
plt.axvline(a0, color = 'k', linestyle = '--', label = 'a\u2080 verdadera')
plt.ylabel('#Cantidad de ocurrencias')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

#%% Histogramas de los estimadores de frecuencia
plt.figure()
plt.suptitle("Histograma del estimador de frecuencia")

plt.subplot(1,2,1)
plt.title("SNR = 10dB") 
plt.hist(estimador_f_R10, label = 'Rectangular',    alpha = trans, bins = 10)
plt.hist(estimador_f_F10, label = 'Flattop',        alpha = trans, bins = 10)
plt.hist(estimador_f_B10, label = 'Blackmanharris', alpha = trans, bins = 10)
plt.hist(estimador_f_H10, label = 'Hamming',        alpha = trans, bins = 10)
plt.axvline(np.mean(omega1), color = 'k', linestyle = '--', label = 'Ω1 verdadera')
# plt.xlabel('Frecuencia [Hz]')
plt.ylabel('# Cantidad de ocurrencias')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.title("SNR = 3dB")
plt.hist(estimador_f_R3, label = 'Rectangular',    alpha = trans, bins = 10)
plt.hist(estimador_f_F3, label = 'Flattop',        alpha = trans, bins = 10)
plt.hist(estimador_f_B3, label = 'Blackmanharris', alpha = trans, bins = 10)
plt.hist(estimador_f_H3, label = 'Hamming',        alpha = trans, bins = 10)
plt.axvline(np.mean(omega1), color = 'k', linestyle = '--', label = 'Ω1 verdadera')
# plt.xlabel('Frecuencia [Hz]')
plt.ylabel('# Cantidad de ocurrencias')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

#%% Efecto del zero padding en el estimador de frecuencia (SNR = 10dB)
N_zp = 2 * N  # 2000 puntos

#%% CONCLUSIONES
# El estimador de energia es muy dependiente de la ventana
# Para todo lo que sea energia, el desparramo de la misma depende de la energia
# En contraposicion, el estimador de la frecuencia no depende de la energia --> forma concava
# Zero padding --> va a bajar la varianza porque estoy midiendo en mas puntos intermedios
# Tengo una mejor resolucion espectral ficticia pero efectiva a la hora de estimar la frecuencia 
# El hecho de agregar ruido, mejora o empeora la estimacion? 
# El ruido va a tener mas relevancia en la estimacion de frecuencia que en el de energia
# El zp se lleva puesto eso 
# na es un proceso gaussiano incorrelado, cuando promediamos eso tiende a cero
# Da lo mismo promediar en las realizaciones que en el tiempo (valor nulo) --> ergodicidad
