# Primero importo la libreria numpy
import numpy as np
import matplotlib.pyplot as plt
import Funciones as ts1

# %% ------------------------------- EJERCICIO 1 -------------------------------
"""
1) Sintetizar y graficar: --> uso funciones de numpy
    a. Una señal sinusoidal de 2KHz
    b. Misma señal amplificada y desfazada en pi/2
    c. Señal anterior modulada en amplitud por otra señal sinusoidal de la mitad de frecuencia. --> modular es multiplicar por otra señal, en este caso va a ser de 1kHz --> sen(2*t) * sen(t) --> reemplazo A (amplitud) por la funcion que modula
    d. Señal anterior recortada al 75% de su amplitud --> potencia = A**2 / 2 --> despues se hace con una funcion de numpy --> la salida es recta arriba y abajo --> recorto todo lo que supere el 75% de la potencia
    --> las funciones con las que trabajamos no son periodicas (vemos N-muestras) --> vamos a calcular energias --> tomamos potencia como energia
    --> tengo que recortar la amplitud al valor del 75% de la potencia de la señal
    --> una manera es ir recorriendo el vector, y cuando llego a un valor que es mayor que el 75% de la potencia, lo cambio por el valor corte = treasure = 75% de la potencia
    --> otra manera, busco la funcion de numpy --> np.clip
    e. Una señal cuadrada de 4KHz
    f. Un pulso rectangular de 10ms
    g. En cada caso indique tiempo entre muestras, numero de mustras y potencia
"""
    
#%% ITEM 1
# ------------------------------- Señal sinusoidal de 2kHz -------------------------------
# Defino mis variables 
amplitud = 1
offset = 0
frecuencia = 2000 # 2kHz = 2000 Hz
fase = 0
N = 200 # Cantidad de muestras
fs = 60000 # Pongo fs > 2 * frecuencia --> Teorema de Nyquist-Shannon 

# Llamo a mi funcion
_,f1 = ts1.mi_funcion_sen(amplitud, offset, frecuencia, fase, N, fs)

# Genero otra ventana para los graficos
plt.figure()  # Tamaño de la figura (ancho, alto)
plt.subplot(2,2,1)

plt.title("Señal Sinusoidal")

# Grafico la señal senoidal de 2KHz
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.plot(_,f1,  label = 'Sinusoidal de 2kHz')
plt.legend()
plt.grid()
plt.show()

# TIEMPO ENTRE MUESTRAS
Ts = 1/fs
print(f"El tiempo entre muestras es {Ts} segundos")

# Calculo la potenacia
x = ts1.calcular_potencia(f1)
print("Potencia de la señal f1: ", x)

#%% ITEM 2
# ------------------------------- Señal sinusoidal de 2kHz, amplificada y desfazada en pi/2 -------------------------------
# Defino mis variables 
amplitud = 2
offset = 0
frecuencia = 2000 # 2kHz = 2000 Hz
fase = np.pi/2
N = 200 # Cantidad de muestras
fs = 60000

# Llamo a mi funcion
_,f2 = ts1.mi_funcion_sen(amplitud, offset, frecuencia, fase, N, fs) 

# Grafico la señal senoidal de 2KHz, pero amplificada y desfazada en pi/2
plt.subplot(2,2,2)
plt.title("Señal Sinusoidal")

# Grafico la señal senoidal de 2KHz
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.plot(_,f1,  label = 'Sinusoidal de 2kHz')
plt.legend()

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
# plt.plot(_,f2, 'x:', label = f'2kHz, amplificada y desfazada en {fase}')
plt.plot(_,f2, label = '2kHz, amplificada y desfazada en π/2')
plt.legend()
plt.grid()
plt.show()

# TIEMPO ENTRE MUESTRAS
Ts = 1/fs
print(f"El tiempo entre muestras es {Ts} segundos")

# Calculo la potenacia
x = ts1.calcular_potencia(f2)
print("Potencia de la señal f2: ", x)

#%% ITEM 3
# ------------------------------- Señal sinusoidal modulada por otra señal sinusoidal de 1kHz -------------------------------
# Defino mis variables
amplitud = 1
offset = 0
frecuencia = 2000 # 2kHz = 2000 Hz
fase = 0
N = 200 # Cantidad de muestras
fs = 60000

# Llamo a mi funcion
_,f3 = ts1.mi_funcion_sen_modulada(amplitud, offset, frecuencia, fase, N, fs)

# Grafico la señal senoidal de 2KHz, pero modulada por otra señal de la mitad de la frecuencia
plt.subplot(2,2,3)
plt.title("Señal Sinusoidal")

# Grafico la señal senoidal de 2KHz
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.plot(_,f1,  label = 'Sinusoidal de 2kHz')
plt.legend()

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.plot(_, f3, label = 'Señal modulada') # Genero el grafico de la señal
plt.legend()
plt.grid()
plt.show()

# TIEMPO ENTRE MUESTRAS
Ts = 1/fs
print(f"El tiempo entre muestras es {Ts} segundos")

# Calculo la potenacia
x = ts1.calcular_potencia(f3)
print("Potencia de la señal f3: ", x)

#%% ITEM 4
# ------------------------------- Misma señal pero recortada al 75% de su potencia -------------------------------
# Defino mis variables
amplitud = 1
offset = 0
frecuencia = 2000 # 2kHz = 2000 Hz
fase = 0
N = 200 # Cantidad de muestras
fs = 60000

# Llamo a mi funcion
_,f4 = ts1.mi_funcion_sen_recortada(amplitud, offset, frecuencia, fase, N, fs)

plt.subplot(2,2,4)
plt.title("Señal Sinusoidal")

# Grafico la señal senoidal de 2KHz
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.plot(_,f1,  label = 'Sinusoidal de 2kHz')
plt.legend()

# Grafico la señal recortada
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.plot(_,f4, label = 'Señal recortada al 75%')
plt.legend()
plt.grid()
plt.show()

# TIEMPO ENTRE MUESTRAS
Ts = 1/fs
print(f"El tiempo entre muestras es {Ts} segundos")

# Calculo la potenacia
x = ts1.calcular_potencia(f4)
print("Potencia de la señal f4: ", x)

#%% ITEM 5
# ------------------------------- Señal cuadrada de 4kHz -------------------------------
# Defino mis variables
frecuencia = 4000
fs = 60000
N = 50 # Cantidad de muestras
offset = 0
fase = 0

# Llamo a mi funcion
_,f5 = ts1.mi_funcion_cuadrada(frecuencia, fs, N, offset, fase) # Comparo dos cosas de valores muy diferentes

# Grafico la señal cuadrada
plt.figure()
plt.subplot(1,2,1)
plt.title('Señal cuadrada de 4kHz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.plot(_,f5, label = 'Señal cuadrada de 4Hz')
plt.legend()
plt.grid()
plt.show()

# TIEMPO ENTRE MUESTRAS
Ts = 1/fs
print(f"El tiempo entre muestras es {Ts} segundos")

# Calculo la potenacia
x = ts1.calcular_potencia(f5)
print("Potencia de la señal f5: ", x)

#%% ITEM 6
# ------------------------------- Pulso rectangular de 10ms -------------------------------
# Defino mis variables
t0 = 1 # Tiempo donde empieza el pulso
tf = 11 # Tiempo donde termina el pulso 
# --> Con este t0 y tf mi pulso rectangular dura 10ms
N = 200 # Cantidad de muestras
h = 1 # Altura del pulso 

# Llamo a mi funcion
f6 = ts1.mi_funcion_pulso(t0, tf, N, h)

# Grafico
plt.subplot(1,2,2)
plt.title('Pulso rectangular de 10ms')
plt.plot(f6, label = 'Pulso rectangular de 10ms')
# otra manera de graficar el pulso: 
# plt.stem(x)
plt.xlabel('Tiempo[ms]')
plt.ylabel('Amplitud [V]')
plt.axis([-2,15,0,1.5]) #Limites del grafico ([Xmin,Xmax,Ymin,Ymax]), si no se agrega este plt (axis) el programa por default busca mostrar todo el grafico completo
plt.legend()
plt.grid()
plt.show()

# TIEMPO ENTRE MUESTRAS
Ts = 1/fs
print(f"El tiempo entre muestras es {Ts} segundos")

# Calculo la energia
x = ts1.calcular_energia(f6)
print("Energia de la señal f6: ", x)

#%% OBSERVACIONES
# Pongo N = 100, porque si pongo N = 1000 el tiempo entre muestras es demasiado chico y no se aprecian las señales en los graficos
# De todas formas, con el N seleccionado tambien el Ts es muy bajo pero no tanto
# --> Esta bien que sea bajo, porque las señales que estamos graficando tienen frecuencias muy altas
# --> Como cumplo Nyquist para evitar aliasing, cuando hago Ts = fs/2, se hace muy chico ese tiempo




