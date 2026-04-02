import numpy as np
import matplotlib.pyplot as plt

# Defino la funcion y los parametros 
N = 8
n = np.arange(N)
x1 = 4 + 3 * np.sin(np.pi/2 * n)
# x = np.sin(np.pi/2 * n)

# DFT
X1 = np.fft.fft(x1)

# Modulo y fase de la DFT
X1_mod = np.abs(X1)
X1_ph = np.angle(X1)

# Graficos
plt.figure()
plt.suptitle("DFT de x[n] con frecuencia = π/2")

# Modulo
plt.subplot(1,2,1)
plt.axhline(0, color='black', linewidth=1.5)  # Eje horizontal y = 0 mas oscuro 
plt.stem(n, X1_mod, basefmt=" ")
plt.xticks(range(N))
plt.title("Modulo de la DFT de x[n]")
plt.ylabel("|X[k]|")
plt.xlabel("k")
plt.grid()

# Fase
plt.subplot(1,2,2)
plt.axhline(0, color='black', linewidth=1.5)  # Eje horizontal y = 0 mas oscuro 
plt.stem(n, X1_ph, basefmt=" ")
plt.title("Fase de la DFT de x[n]")
plt.ylabel("∠X[k]")
plt.xlabel("k")
plt.grid()

plt.show()

#%% ESPECTRO DE POTENCIA 

Potencia1 = (X1_mod**2) / (N**2)

# Graficos
plt.figure()
plt.title("Espectro de potencia")

plt.axhline(0, color='black', linewidth=1.5)  # Eje horizontal y = 0 mas oscuro 
plt.stem(n, Potencia1, basefmt=" ")
plt.xticks(range(N))
plt.ylabel("Potencia")
plt.xlabel("k")
plt.grid()

plt.show()

#%% Conclusiones
# La potencia de la señal es A^2 / 2 --> Potencia = 9/2 = 4.5 
# Pero como tengo dos deltas, la potencia se reparte --> los palitos tienen que estar a una altura de 4.5/2 = 2.25

# Viendo los graficos, observamos que la mayor cantidad de energia se centra en el cero, ese aporte corresponde a la contsante 4 de la secuencia.
# Tambien vemos energia en k = 2 y k = 6, eso se debe a la senoidal. --> esa es la parte de la potencia que aparece repartida entre esos puntos.

#%% Ahora pruebo con otra frecuencia 3/2pi

x2 = 4 + 3 * np.sin(3 * np.pi/2 * n)

# DFT
X2 = np.fft.fft(x2)

# Modulo y fase de la DFT
X2_mod = np.abs(X2)
X2_ph = np.angle(X2)

# Graficos
plt.figure()
plt.suptitle("DFT de x[n] con frecuencia = π/3")

# Modulo
plt.subplot(1,2,1)
plt.axhline(0, color='black', linewidth=1.5)  # Eje horizontal y = 0 mas oscuro 
plt.stem(n, X2_mod, basefmt=" ")
plt.xticks(range(N))
plt.title("Modulo de la DFT de x[n]")
plt.ylabel("|X[k]|")
plt.xlabel("k")
plt.grid()

# Fase
plt.subplot(1,2,2)
plt.axhline(0, color='black', linewidth=1.5)  # Eje horizontal y = 0 mas oscuro 
plt.stem(n, X2_ph, basefmt=" ")
plt.title("Fase de la DFT de x[n]")
plt.ylabel("∠X[k]")
plt.xlabel("k")
plt.grid()

plt.show()

#%% ESPECTRO DE POTENCIA 

Potencia2 = (X2_mod**2) / (N**2)

# Graficos
plt.figure()
plt.title("Espectro de potencia")

plt.axhline(0, color='black', linewidth=1.5)  # Eje horizontal y = 0 mas oscuro 
plt.stem(n, Potencia2, basefmt=" ")
plt.xticks(range(N))
plt.ylabel("Potencia")
plt.xlabel("k")
plt.grid()

plt.show()

#%% Ahora pruebo con otra frecuencia 1/3pi

x3 = 4 + 3 * np.sin(np.pi/3 * n)

# DFT
X3 = np.fft.fft(x3)

# Modulo y fase de la DFT
X3_mod = np.abs(X3)
X3_ph = np.angle(X3)

# Graficos
plt.figure()
plt.suptitle("DFT de x[n] con frecuencia = π/3")

# Modulo
plt.subplot(1,2,1)
plt.axhline(0, color='black', linewidth=1.5)  # Eje horizontal y = 0 mas oscuro 
plt.stem(n, X3_mod, basefmt=" ")
plt.xticks(range(N))
plt.title("Modulo de la DFT de x[n]")
plt.ylabel("|X[k]|")
plt.xlabel("k")
plt.grid()

# Fase
plt.subplot(1,2,2)
plt.axhline(0, color='black', linewidth=1.5)  # Eje horizontal y = 0 mas oscuro 
plt.stem(n, X3_ph, basefmt=" ")
plt.title("Fase de la DFT de x[n]")
plt.ylabel("∠X[k]")
plt.xlabel("k")
plt.grid()

plt.show()

#%% Espectro de la potencia
Potencia3 = (X3_mod**2) / (N**2)

# Graficos
plt.figure()
plt.title("Espectro de potencia")

plt.axhline(0, color='black', linewidth=1.5)  # Eje horizontal y = 0 mas oscuro 
plt.stem(n, Potencia3, basefmt=" ")
plt.xticks(range(N))
plt.ylabel("Potencia")
plt.xlabel("k")
plt.grid()

plt.show()

#%% Conclusiones

# Cuando la fase de la señal no cae exactamente en un bin, cuando quiero graficar el espectro de potencia, el espectro de desparrama, es decir hay derrame espectral
# Derramo espetral = leakage
 



