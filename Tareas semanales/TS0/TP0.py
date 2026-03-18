#%% Primero importo las librerias
import numpy as np
import matplotlib.pyplot as plt

# Defino mi funcion
def mi_funcion_sen(amplitud = 1, offset = 0, frecuencia = 1, fase = 0, N = 1000, fs = 1000):
    """
    - amplitud: es la amplitud maxima. [amplitud] = [V]
    - offset: es mi amplitud media. [offset] = [V]
    - frecuencia: es la frecuencia de la señal. [frecuencia] = [Hz]
    - fase: es la fase inicial. [fase] = [rad]
    - N: es la cantidad de muestras a generar
    - fs: es la frecuencia de muestreo --> cantidad de muestras que se toman cada 1 segundo. [fs] = [Hz]
    """
    Ts = 1/fs # Es el tiempo en el cual se toma cada muestra

    tt = np.arange(start = 0, stop= N*Ts, step = Ts)

    xx = amplitud * np.sin(2 * np.pi * frecuencia * tt + fase) + offset

    return tt, xx


# Defino mis variables
N = 1000
amplitud = 2
offset = 1
frecuencia = 5
fase = np.pi/4
fs = 1000

# Llamo a mi funcion
tt, xx = mi_funcion_sen(amplitud, offset, frecuencia, fase, N, fs)

# Grafico la señal
plt.title("Señal Senoidal Generada con una frecuencia de 5Hz")
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.plot(tt, xx, linestyle = '-', color = 'r' ) # Genero el grafico de la señal con linea 'continua' de color 'rojo'

#%% --------------------------------------- ITEM 1 (EJERCICIO BONUS) ---------------------------------------
# Genero otra ventana para los graficos
plt.figure(figsize=(10, 6))  # Tamaño de la figura (ancho, alto)

# Señal 500 Hz, estoy exactamente en Nyquist --> f = 2fs
plt.subplot(2, 2, 1)
tt, xx = mi_funcion_sen(1, 0, 500, 0, 1000, 1000)
plt.plot(tt, xx, '-o', color='blue', markersize=1)
plt.title("Señal Senoidal con una frecuencia de 500 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)

# Señal 999 Hz
plt.subplot(2, 2, 2)
tt, xx = mi_funcion_sen(1, 0, 999, 0, 1000, 1000)
plt.plot(tt, xx, '-o', color='green', markersize=1)
plt.title("Señal Senoidal con una frecuencia de 999 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)

# Señal 1001 Hz
plt.subplot(2, 2, 3)
tt, xx = mi_funcion_sen(1, 0, 1001, 0, 1000, 1000)
plt.plot(tt, xx, '-o', color='orange', markersize=1)
plt.title("Señal Senoidal con una frecuencia de 1001 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)

# Señal 2001 Hz
plt.subplot(2, 2, 4)
tt, xx = mi_funcion_sen(1, 0, 2001, 0, 1000, 1000)
plt.plot(tt, xx, '-o', color='grey', markersize=1)
plt.title("Señal Senoidal con una frecuencia de 2001 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)

plt.tight_layout()  
plt.show()

#%% --------------------------------------- ITEM 2 (EJERCICIO BONUS) ---------------------------------------

# --------------------------------------- Grafico una señal cuadrada ---------------------------------------

from scipy import signal

def mi_funcion_cuadrada (frecuencia, fs, N, offset, fase):
    Ts = 1/fs # Es el tiempo en el cual se toma cada muestra

    ttc = np.arange(start = 0, stop= N*Ts, step = Ts)

    xxc = signal.square(2 * np.pi * frecuencia * ttc + fase) + offset

    return ttc, xxc
    
# Defino mis variables
N = 100
offset = 0
frecuencia = 4
fase = 0
fs = 100

# Llamo a mi funcion
ttc, xxc = mi_funcion_cuadrada(frecuencia, fs, N, offset, fase)

# Graficamos la señal cuadrada generada
plt.figure(figsize=(10, 4))
plt.plot(ttc, xxc, label='Señal Cuadrada')
plt.title('Generación de Señal Cuadrada con una frecuencia de 4Hz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.show()

"""
OTRA MANERA DE GRAFICAR UNA SEÑAL CUADRADA USANDO scipy

from scipy.signal import sweet_poly
# Parámetros de la señal
amplitud = 1 # Volts
offset = 0 
frecuencia = 50 # Hz
fase = 0 # radianes
tiempo_total = 0.05 # segundos
fs = 5000 # Frecuencia de muestreo (Hz)
t = np.arange(0, tiempo_total, 1/fs)

# Definimos el polinomio que describe la frecuencia instantánea
# En este caso, una aproximación simple con una frecuencia constante
polinomio = np.poly1d([frecuencia])

# Generamos la señal utilizando sweep_poly
senal_senoidal = sweep_poly(t, polinomio)

# Aproximamos la señal cuadrada utilizando una combinación de senos
senal_cuadrada = np.sign(senal_senoidal)

# Graficamos la señal cuadrada generada
plt.figure(figsize=(10, 4))
plt.plot(t, senal_cuadrada, label='Señal Cuadrada Aproximada')
plt.title('Generación de Señal Cuadrada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.show()

"""

# --------------------------------------- Grafico el impulso unitario ---------------------------------------
# Defino mi funcion impulso unitario
def mi_impulso(posicion = 0, N = 21, fs = 1):
    """
    posicion: lugar donde aparece el impulso
    N: cantidad de muestras, numero de muestras que dura el impulso --> controla la longitud de la secuencia
    fs: frecuencia de muestreo
    """

    Ts = 1/fs

    tt = np.arange(start = 0, stop = N*Ts, step = Ts)

    # Genero el impulso con la funcion de scipy
    xx = signal.unit_impulse(N, idx = posicion) 

    return tt, xx


# Defino variables
N = 21
fs = 1
posicion = 10

# Llamo a la funcion
tt, xx = mi_impulso(posicion, N, fs)

# Grafico
plt.figure(figsize=(10, 4))
plt.title("Impulso Unitario δ[n]")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")

plt.stem (tt, xx)
plt.grid(True)

plt.show()


