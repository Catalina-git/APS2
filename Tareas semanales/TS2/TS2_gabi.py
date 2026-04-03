import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import Funciones as ts2

fs = 2 # Hz, frecuencia de muestreo
f0 = 20 # Hz, frecuencia de la senoidal
N = 1000 # Muestras por ciclo
vmax = 1 # Volts
dc = 0 # Valor medio, volts
Ts = 1/fs
df = fs/N

tt, xx = ts2.mi_funcion_sen(vmax, dc, f0, N, fs)
plt.plot(tt, xx)

#%% Potencia de señal
#Fijamos en 1 Watt el valor de potencia de la señal para despejar Pr
#Queda SNR = 10* Log(1/Pr) y con eso regulo cuanto necesito de Pr para el
#SNR que busco

mu = 0
vmax = np.sqrt(2)
tt, xx = ts2.mi_funcion_sen(vmax, dc, f0, N, fs)
Px = np.var(xx)
SNR = 40
Pr = 10**(-SNR/10)

U_n = np.random.normal(mu,np.sqrt(Pr), N)

xxn = xx + U_n

plt.figure()
plt.plot(tt, xxn, label =f'Senoidal con Ruido - SNR = {SNR}')
plt.plot(tt, xx, 'r', label = 'Senoidal pura')
plt.grid()
plt.legend()
plt.show()

#%% Cuantización

B = 4  # bits
Vfs = 2  # volts
qq = Vfs / (2**B)
#Pr = qq**2/12

# Cuantización
xxq_idx = np.round(xxn / qq)
xxq = xxq_idx * qq

# Error
#error = xxn - xxq


#%%

XXQ = np.fft.fft(xxq)/N
XXQ = XXQ[:N//2]
XXQ_db = 20 * np.log(np.abs(XXQ)*2)

freq = np.fft.fftfreq(N, Ts)
freq = freq[:N//2]

plt.figure()
plt.title(f'Densidad espectral de potencia - SNR = {SNR}')
plt.plot(freq, XXQ_db, label = 'Señal cuantizada con ruido')
plt.xlabel('Frecuencia')
plt.ylabel('dB')
plt.grid()
plt.show()


