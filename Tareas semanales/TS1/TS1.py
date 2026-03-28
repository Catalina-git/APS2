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
plt.plot(_,f1)
plt.grid()
plt.show()

# TIEMPO ENTRE MUESTRAS
Ts = 1/fs
print(f"El tiempo entre muestras es {Ts} segundos")

# Calculo la potenacia
x = ts1.calcular_potencia(f1)
print("Potencia de la señal f1: ", x)

#%% ITEM 2
# ------------------------------- Señal sinusoidal de 2kHz, amplificada 3dB y desfazada en pi/2 -------------------------------
# Defino mis variables 
amplitud = 10**(3/20) # sale de la formula --> 3dB = 20*log(Amodulada/Aoriginal) 
offset = 0
frecuencia = 2000 # 2kHz = 2000 Hz
fase = np.pi/2
N = 200 # Cantidad de muestras
fs = 60000

# Llamo a mi funcion
_,f2 = ts1.mi_funcion_sen(amplitud, offset, frecuencia, fase, N, fs) 

# Grafico la señal senoidal de 2KHz, pero amplificada y desfazada en pi/2
plt.subplot(2,2,2)
plt.title("Señal Sinusoidal amplificada 3dB")

# Grafico la señal senoidal de 2KHz
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.plot(_,f1,  label = 'Sinusoidal de 2kHz')
plt.legend()

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
# plt.plot(_,f2, 'x:', label = f'2kHz, amplificada y desfazada en {fase}')
plt.plot(_,f2, label = '2kHz, amplificada 3dB y desfazada en π/2')
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
# ------------------------------- Señal sinusoidal modulada por otra señal sinusoidal de 1000kHz -------------------------------
# Defino mis variables
amplitud = 1
offset = 0
frecuencia = 1000 # 2kHz = 2000 Hz
fase = 0
N = 200 # Cantidad de muestras
fs = 60000

# Llamo a mi funcion
_,f3 = ts1.mi_funcion_sen_modulada(amplitud, offset, frecuencia, fase, N, fs)

# Grafico la señal senoidal de 2KHz, pero modulada por otra señal de la mitad de la frecuencia
plt.subplot(2,2,3)
plt.title("Señal Sinusoidal modulada en amplitud")

# Grafico la señal senoidal de 2KHz
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.plot(_,f1,  label = 'Sinusoidal de 2kHz')
plt.legend()

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.plot(_, f3, label = 'Señal modulada en amplitud') # Genero el grafico de la señal
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
plt.title("Señal Sinusoidal saturada al 75%")

# Grafico la señal senoidal de 2KHz
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.plot(_,f1,  label = 'Sinusoidal de 2kHz')
plt.legend()

# Grafico la señal recortada
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.plot(_,f4, label = 'Señal saturada al 75%')
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
plt.plot(_,f5)
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
plt.plot(f6)
# otra manera de graficar el pulso: 
# plt.stem(x)
plt.xlabel('Tiempo[ms]')
plt.ylabel('Amplitud [V]')
plt.axis([-2,15,0,1.5]) #Limites del grafico ([Xmin,Xmax,Ymin,Ymax]), si no se agrega este plt (axis) el programa por default busca mostrar todo el grafico completo
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

# %% ------------------------------- EJERCICIO 2 -------------------------------

"""

2) Dado h[n] = delta[n] - delta[n - 4], entontrar y[n] = x[n] * h[n] para cada una de las siguientes x[n]:
    a) x[n] = cos(w0 * n * Ts). Expresar la respuesta como un unico coseno de la forma A * cos(w0 * n * Ts + phi).
    b) x[n] = (1/2)^(n) * u[n]
    c) x[n] = u[n + 1] - u[n - 2]
"""

# VERIFICO CONVOLUCION 
#%% ITEM b
# Rango de n
n = np.arange(-10, 20)

# Defino delta
def delta(n):
    return np.where(n == 0, 1, 0)

# Defino h[n]
h = delta(n) - delta(n - 4)

# a) x[n] = cos(w0 * n * Ts)
# Elijo valores de w0 y de Ts para poder graficar
w0 = 0.5
Ts = 1

xa = np.cos(w0 * n * Ts)

ya = np.convolve(xa, h, mode = 'same') # --> el modo 'same' me va a devolver una version centrada de la convolucion, es decir introduce un desplazamiento temporal

# Este es el resultado que obtuve calculando el producto de convolucion a mano
A = -2 * np.sin(2 * w0 * Ts) 
phi = - 2 * w0 * Ts - np.pi/2
ya_analitica = A * np.cos(w0 * n * Ts + phi)

# Grafico
plt.figure(figsize = (8, 10))

plt.suptitle("Verificación de la convolución entre h[n] = δ[n] - δ[n - 4] y las distintas x[n] dadas vs resultado analítico")
plt.subplot(3, 1, 1)
#Verificación convolución vs resultado analítico
# x[n]
plt.stem(n, xa, linefmt='C0-', markerfmt='C0o', basefmt=" ", label='x[n] = cos(ω₀ * n * Ts)')

# y[n] convolución
plt.stem(n, ya, linefmt='C1-', markerfmt='C1s', basefmt=" ", label='y[n] (convolución)')

# y[n] analítica
plt.stem(n, ya_analitica, linefmt='C2-', markerfmt='C2^', basefmt=" ", label='y[n] (analítica)')

# Líneas suaves
plt.plot(n, ya, 'C1--', alpha=0.6)
plt.plot(n, ya_analitica, 'C2--', alpha=0.8)

plt.title("Item a")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()

#%% ITEM b
# b) x[n] = (1/2)^(n) * u[n]
# u[n] = 1 si n >= 0
# u[n] = 0 si n < 0
 
# Defino la funcion u[n]
def u(n):
    return np.where(n >= 0, 1, 0)

xb = (1/2)**n * u(n)

yb = np.convolve(xb, h, mode = 'same')

# Este es el resultado que obtuve calculando el producto de convolucion a mano
yb_analitica = (1/2)**n * u(n) - (1/2)**(n - 4) * u(n - 4)

# Grafico
plt.subplot(3, 1, 2)

# x[n]
plt.stem(n, xb, linefmt='C0-', markerfmt='C0o', basefmt=" ", label='x[n] = (1/2)ⁿ u[n]')

# y[n] convolución
plt.stem(n, yb, linefmt='C1-', markerfmt='C1s', basefmt=" ", label='y[n] (convolución)')

# y[n] analítica
plt.stem(n, yb_analitica, linefmt='C2-', markerfmt='C2^', basefmt=" ", label='y[n] (analítica)')

# Líneas suaves
plt.plot(n, yb, 'C1--', alpha=0.6)
plt.plot(n, yb_analitica, 'C2--', alpha=0.8)

plt.title("Item b")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()

#%% ITEM C
# c) x[n] = u[n + 1] - u[n - 2]
# u[n + 1] = 1 si n >= -1 ; 0 si n < -1
# u[n- 2] = 1 si n >= 2 ; 0 si n < 2

xc = u(n + 1) - u(n - 2)

yc = np.convolve(xc, h, mode = 'same')

# Este es el resultado que obtuve calculando el producto de convolucion a mano
yc_analitica = u(n + 1) - u(n - 2) - u(n - 3) + u(n - 6)

# Grafico
plt.subplot(3, 1, 3)

# x[n]
plt.stem(n, xc, linefmt='C0-', markerfmt='C0o', basefmt=" ", label='x[n] = u[n + 1] - u[n - 2]')

# y[n] convolución
plt.stem(n, yc, linefmt='C1-', markerfmt='C1s', basefmt=" ", label='y[n] (convolución)')

# y[n] analítica
plt.stem(n, yc_analitica, linefmt='C2-', markerfmt='C2^', basefmt=" ", label='y[n] (analítica)')

# Líneas suaves
plt.plot(n, yc, 'C1--', alpha=0.6)
plt.plot(n, yc_analitica, 'C2--', alpha=0.8)

plt.title("Item c")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()

#%% Observaciones sobre stem 

# linefmt = ... --> Es el formato de las lineas verticales
# 'C2-' --> Verde y linea continua
# 'C0-' --> Azul y linea continua
# 'C1-' --> Naranje y linea continua

# markerfmt = ... --> Es el formato de los marcadores (puntitos de arriba) 
# 'C2^' --> Verde (el codigo se colores es siempre el mismo) y la forma del marcador es un triangulo 
# '^' --> triángulo
# 'o' --> círculo
# 's' --> cuadrado 
# 'd' --> rombo

# basefmt = ... --> Es la linea base (horizontal)
# " " (espacio) --> no dibuja la base
# Si no pongo el basefmt me dibuja una línea horizontal en y = 0





