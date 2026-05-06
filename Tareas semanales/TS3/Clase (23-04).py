#%% LLamo a las librerias que se van a utilizar
import numpy as np
import matplotlib.pyplot as plt

# Parametros
fs = 1000 
N = 1000
k0 = N/4
f0 = k0 * fs/N 
ts = 1/fs
tt = np.arange(N) * ts
df = fs/N # Resolucion temporal

# Eje de frecuencias
freqs = np.arange(0, fs, df)

#%% k0 = N/4
# Señal (potencia unitaria)
vmax = np.sqrt(2)
dc = 0
x = dc + vmax * np.sin(2*np.pi*f0*tt)

# FFT
X = np.fft.fft(x)
Xmod = (np.abs(X) / N)**2
Xmod_db = 10 * np.log10(Xmod * 2 + 1e-12)

# Fase
X_fase = np.angle(X)

# Eje de frecuencias
freqs = np.arange(0, fs, df)

# Graficos
plt.figure()
plt.suptitle("Senoidal con frecuencia N/4 ")

plt.subplot(2, 1, 1)
plt.title("Magnitud")
plt.plot(freqs, Xmod_db, marker = 'o', markersize = 2)
plt.xlim([0, fs/2]) # Es un espectro par 
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.title("Fase")
plt.plot(freqs, X_fase, marker = 'o', markersize = 2)
plt.xlim([0, fs/2]) # Es un espectro par 
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Fase [rad]')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

#%% k0 = N/4 + 0.25

k0 = N/4 + 0.25
f0 = k0 * fs/N 
ts = 1/fs
tt = np.arange(N) * ts

# Señal (potencia unitaria)
vmax = np.sqrt(2)
dc = 0
x1 = dc + vmax * np.sin(2*np.pi*f0*tt)

# FFT
X1 = np.fft.fft(x1)
Xmod1 = (np.abs(X1) / N)**2
Xmod_db1 = 10 * np.log10(Xmod1 * 2 + 1e-12)

# Fase
X_fase1 = np.angle(X1)

# Graficos
plt.figure()
plt.suptitle("Senoidal con frecuencia N/4 + 0.25 ")

plt.subplot(2, 1, 1)
plt.title("Magnitud")
plt.plot(freqs, Xmod_db1, marker = 'o', markersize = 2)
plt.xlim([0, fs/2]) # Es un espectro par 
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.title("Fase")
plt.plot(freqs, X_fase1, marker = 'o', markersize = 2)
plt.xlim([0, fs/2]) # Es un espectro par 
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Fase [rad]')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()


#%% k0 = N/4 + 0.5

k0 = N/4 + 0.5
f0 = k0 * fs/N 
ts = 1/fs
tt = np.arange(N) * ts

# Señal (potencia unitaria)
vmax = np.sqrt(2)
dc = 0
x2 = dc + vmax * np.sin(2*np.pi*f0*tt)

# FFT
X2 = np.fft.fft(x2)
Xmod2 = (np.abs(X2) / N)**2
Xmod_db2 = 10 * np.log10(Xmod2 * 2 + 1e-12)

# Fase
X_fase2 = np.angle(X2)

# Graficos
plt.figure()
plt.suptitle("Senoidal con frecuencia N/4 + 0.5 ")

plt.subplot(2, 1, 1)
plt.title("Magnitud")
plt.plot(freqs, Xmod_db2, marker = 'o', markersize = 2)
plt.xlim([0, fs/2]) # Es un espectro par 
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.title("Fase")
plt.plot(freqs, X_fase2, marker = 'o', markersize = 2)
plt.xlim([0, fs/2]) # Es un espectro par 
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Fase [rad]')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

#%% k0 = N/4 + 1

k0 = N/4 + 1
f0 = k0 * fs/N 
ts = 1/fs
tt = np.arange(N) * ts

# Señal (potencia unitaria)
vmax = np.sqrt(2)
dc = 0
x3 = dc + vmax * np.sin(2*np.pi*f0*tt)

# FFT
X3 = np.fft.fft(x3)
Xmod3 = (np.abs(X3) / N)
Xmod_db3 = 10 * np.log10(Xmod3 * 2 + 1e-12)

# Fase
X_fase3 = np.angle(X3)

# Graficos
plt.figure()
plt.suptitle("Senoidal con frecuencia N/4 + 1")

plt.subplot(2, 1, 1)
plt.title("Magnitud")
plt.plot(freqs, Xmod_db3, marker = 'o', markersize = 2)
plt.xlim([0, fs/2]) # Es un espectro par 
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.title("Fase")
plt.plot(freqs, X_fase3, marker = 'o', markersize = 2)
plt.xlim([0, fs/2]) # Es un espectro par 
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Fase [rad]')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

#%% Graficos del modulo
"""
plt.figure()

plt.plot(freqs, Xmod_db, label = "Frecuencia = N/4", marker = 'o', markersize = 2)
plt.plot(freqs, Xmod_db1, label = "Frecuencia = N/4 + 0.25", marker = 'o', markersize = 2)
plt.plot(freqs, Xmod_db2, label = "Frecuencia = N/4 + 0.5", marker = 'o', markersize = 2)
plt.plot(freqs, Xmod_db3, label = "Frecuencia = N/4 + 1", marker = 'o', markersize = 2)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.xlim([0, fs/2]) # Es un espectro par 
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.title("Espectro de potencia de la DFT de una senoridal con distintas frecuencias")

plt.grid()
plt.legend()
plt.show()

"""

#%% ZERO PADDING
# Vector de ceros de tamaño 9 * N
cant_zp = 9
zp = np.zeros(cant_zp * N) 

# La funcion "concatenate" une dos secuencias en una sola
# --> La uso para crear la cajita (ventana) que vale 1 hasta N - 1, y cero desde ahi hasta (2N - 1)
xp = np.concatenate((x, zp)) 
xp1 = np.concatenate((x1, zp))
xp2 = np.concatenate((x2, zp))
xp3 = np.concatenate((x3, zp))

# muestras_zp = cant_zp * N + N # Estoy agregando ceros para que la señal me quede de longitud 10N

# Calculo la FFT
Xp = np.fft.fft(xp) / N
Xp1 = np.fft.fft(xp1) / N
Xp2 = np.fft.fft(xp2) / N
Xp3 = np.fft.fft(xp3) / N

# Modulos
Xp_mod = np.abs(Xp)**2
Xp_mod1 = np.abs(Xp1)**2
Xp_mod2 = np.abs(Xp2)**2
Xp_mod3 = np.abs(Xp3)**2

# FFT en dB
Xp_dB = 10 * np.log10(Xp_mod)
Xp_dB1 = 10 * np.log10(Xp_mod1)
Xp_dB2 = 10 * np.log10(Xp_mod2)
Xp_dB3 = 10 * np.log10(Xp_mod3)

# Fase 
Xp_fase = np.angle(Xp)

# Ejes de frecuencia
Npadding = len(xp)
df_padding = fs / Npadding
ff_padding = np.arange(Npadding) * df_padding 

# Grafico en db
plt.figure()
plt.suptitle("Zero Padding") # DEP = Densidad Espectral de Potencia --> PSD

# frecuencia = N/4
plt.subplot(2, 2, 1)
plt.title("frecuencia = N/4")
plt.plot(freqs, Xmod_db, marker = 'o', markersize = 2, label = "sin zero padding", linestyle='none')
plt.plot(ff_padding, Xp_dB, 'o', markersize = 2, label = 'con zero padding')
plt.xlim(0, fs/2)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("DEP [db]")
plt.legend()
plt.grid()

# frecuencia = N/4 + 0.25
plt.subplot(2, 2, 2)
plt.title("frecuencia = N/4 + 0.25")
plt.plot(freqs, Xmod_db1, marker = 'o', markersize = 2, label = "sin zero padding", linestyle='none')
plt.plot(ff_padding, Xp_dB1, 'x', markersize = 2, label = 'con zero padding')
plt.xlim(0, fs/2)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("DEP [db]")
plt.legend()
plt.grid()

# frecuencia = N/4 + 0.5
plt.subplot(2, 2, 3)
plt.title("frecuencia = N/4 + 0.5")
plt.plot(freqs, Xmod_db2, marker = 'o', markersize = 2, label = "sin zero padding", linestyle='none')
plt.plot(ff_padding, Xp_dB2, '+', markersize = 2, label = 'con zero padding')
plt.xlim(0, fs/2)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("DEP [db]")
plt.legend()
plt.grid()

# frecuencia = N/4 + 1
plt.subplot(2, 2, 4)
plt.title("frecuencia = N/4 + 1")
plt.plot(freqs, Xmod_db3, marker = 'o', markersize = 2, label = "sin zero padding", linestyle='none')
plt.plot(ff_padding, Xp_dB3, 'o', markersize = 2, label = 'con zero padding')
plt.xlim(0, fs/2)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("DEP [db]")
plt.legend()
plt.grid()

plt.show()

#%% Anotaciones 
# ZERO PADDING
# Esto es hacer una interpolacion, estoy interpolando con una sinc
# La interpolante siempre va a ser una sinc, a menos que le pida que me interpole con otra funcion
# Entonces veo con mejor resolucion la sinc, pero no agrego informacion
# Si tengo dos senoidales muy proximas y las quiero diferenciar
# la unica forma de hacerlo es que las sincs que se van a superponer a estas senoidales esten lo suficientemente separadas para identificar cada uno de los lobulos de las sincs
# Para hacer eso tengo que aumentar la cantidad de tiempo en el que mido
# es decir aumentando el N * Ts
# si no voy a variar la fs, solo me queda aumentar N, y asi voy a poder diferenciar dos señales muy cercanas. 
# Para evidenciar la discontinuidad de la pendiente, voy a necesitar una resolucion lo suficientemente fina que me permita verlo 



