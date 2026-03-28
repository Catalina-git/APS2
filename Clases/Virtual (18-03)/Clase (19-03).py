import numpy as np
import matplotlib.pyplot as plt

fs = 1000 #Hz, frecuencia de muestreo
f0 = 1 #Hz, frecuencia de la senoidal
n = 1000 #muestras por ciclo
vmax = 1 #volts
dc = 0 #valor medio, volts
ts = 1/fs

def mi_senoidal(vmax, dc, f0, n, fs, ph=0):
    
    tt = np.linspace(0, (n-1)*ts, n)
    xx = dc + vmax*np.sin(2*np.pi*f0*tt+ph)
    
    return tt, xx

tt, xx = mi_senoidal(vmax, dc, f0, n, fs, ph=0)
plt.figure()
plt.plot(tt, xx, label = 'Senoidal original')
plt.legend()

#%% Señal con ruido
#Agregamos una señal aleatoria (ruido) a la señal pura

sigma = 1
mu = 0
SNR = 30
Pr = 10**(-SNR/10)
U_n = np.random.normal(mu,np.sqrt(Pr),n)
# U_n = np.random.normal(mu,sigma,n)

xxn = xx + U_n

plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.plot(tt, xxn, label ='Senoidal con Ruido')
plt.grid()
plt.legend()
plt.show()
#%% CUANTIZACION

B = 4 # Cantidad de bits
Vfs = 3 # [V] --> es el voltaje en full scale
# --> asumo que es simetrico, en este caso como es 3 tengo 1.5 para arriba y 1.5 para abajo
qq = Vfs / 2**B# Rango de cuantizacion --> qq = 0,1875

# Para cuantizar tengo que truncar o redondear
xxq = np.round(xx / qq) * qq # Esta cuantizada y codificada
# Lo multiplico devuelta por qq para que me coincida con las unidades de xx

plt.figure(figsize=(8,6))
plt.subplot(2,1,1)
plt.plot(xx, label='Señal original')
plt.plot(xxq, label='Señal cuantizada')
plt.grid(True)
plt.legend()
plt.title("Cuantización")

# Calculo la secuencia de error
# El error de cuntizacion queda acotado entre [-q/2; +q/2] --> en este caso [-0,1875; +0,1875]
# Para calcular el error xx y xxq tienen que estar en la misma magnitud, en lass mismas unidades
# --> Entonces multiplico devuelta por q, no se va a simplificar porque es una operacion no lineal
e = xxq - xx # Es el error de cuantizacion

# Grafico el error 
plt.subplot(2,1,2)
plt.plot(tt, e, label = 'Error de cuantizacion')
plt.axhline( qq/2, color='red',   linewidth=0.8, linestyle='--', label=f'+q/2 = +{qq/2:.4f}V')
plt.axhline(-qq/2, color='red',   linewidth=0.8, linestyle='--', label=f'-q/2 = -{qq/2:.4f}V')
plt.legend()
plt.grid(True)

plt.xlabel("Muestras")
plt.ylabel("Amplitud [V]")

plt.tight_layout()
plt.show()

# --> En este caso tenemos algo correlado
# Hay mucho ordenamiento temporal 

#%% Ahora cuantizamos xxn que tiene un poco de ruido (20db de SNR)

xxq = np.round(xxn / qq) * qq

plt.figure()
plt.subplot(2,1,1)
plt.plot(xxn, label = 'Señal original con ruido')
plt.plot(xxq, label = 'Señal cuantizada')
plt.grid(True)
plt.legend()
plt.title("Cuantización con ruido")

# Calculo la secuencia de error
# El error de cuntizacion queda acotado entre [-q/2; +q/2] --> en este caso [-0,1875; +0,1875]
# Para calcular el error xx y xxq tienen que estar en la misma magnitud, en lass mismas unidades
# --> Entonces multiplico devuelta por q, no se va a simplificar porque es una operacion no lineal
e = xxq - xxn # Es el error de cuantizacion

# Grafico el error 
plt.subplot(2,1,2)
plt.plot(tt, e, label = 'Error de cuantizacion')
plt.axhline( qq/2, color='red',   linewidth=0.8, linestyle='--', label=f'+q/2 = +{qq/2:.4f}V')
plt.axhline(-qq/2, color='red',   linewidth=0.8, linestyle='--', label=f'-q/2 = -{qq/2:.4f}V')
plt.legend()
plt.grid(True)

plt.xlabel("Muestras")
plt.ylabel("Amplitud [V]")

plt.tight_layout()
plt.show()

# --> En este caso tenemos algo fuertemente incorrelado, que tiene que ver con la potencia del ruido
# Es decir que se "rompe" la correlacion --> eso es porque cambiamos Pr (la potencia), entonces hay mas energia, y hay exceso de ruido
# --> Tiene que ver con la cantidad de bits
# Cuando llego a los limites de cuantizacion, hace que se generen cambios de bits, cambios de niveles
# --> Para romper con eso tiene que ser comparable con la mitad del paso de cuantizacion

# La potencia del ruido es tan grande que hace que cuando lo queres representar se pasa al nivel siguiente de cuantizacion
#%% SEÑAL CON RUIDO 

# HISTOGRAMA 
# --> es la funcion distribucion estocastica, es un estimador de la funcion distribucion de probabilidad
# Es una aproximacion a la funcion teorica (cajita) 

plt.figure(figsize=(10,6))

plt.subplot(1,2,1)
plt.hist(e, bins = 30, color = 'steelblue', density = True, edgecolor = 'dimgray', linewidth = 0.8) # La cantidad de bins me da mas rayitas, las cajitas son mas angostas. Si es muy angosta me va a caer en cero porque es muy angostita la cajita
plt.axvline(qq/2, color='red',   linewidth=0.8, linestyle='--', label=f'+q/2 = +{qq/2:.4f}V')
plt.axvline(-qq/2, color='red',   linewidth=0.8, linestyle='--', label=f'-q/2 = -{qq/2:.4f}V')
plt.axhline(1/qq, color='darkorange',   linewidth=0.8, linestyle='--', label=f'1/q = {1/qq:.4f}V')
plt.grid(True)
plt.legend()
plt.title("Histograma del error de cuantización (con ruido)")
plt.xlabel("Error [V]")
plt.ylabel("Densidad")

# AUTOCORRELACION
plt.subplot(1,2,2)

r = np.correlate(e, e, mode = 'full')
lags = np.arange(-len(e)+1, len(e))
L = 50 # Cantidad de lags que quiero ver

plt.plot(lags, r, label = 'Autocorrelacion', color = 'seagreen')
plt.xlim(-L, L)
plt.axvline(0, color='red',   linewidth=0.8, linestyle='--', label='Lag = 0')
plt.legend()
plt.grid(True)
plt.title("Autocorrelación del error (con rudio)")
plt.xlabel("Retardo (lag)")
plt.ylabel("Autocorrelación")

#%% SEÑAL SIN RUIDO 

e = xxq - xx

# HISTOGRAMA
plt.figure(figsize=(10,6))

plt.subplot(1,2,1)
plt.hist(e, bins = 30, color = 'steelblue', density = True, edgecolor = 'dimgray', linewidth = 0.8) # La cantidad de bins me da mas rayitas, las cajitas son mas angostas. Si es muy angosta me va a caer en cero porque es muy angostita la cajita
plt.axvline(qq/2, color='red',   linewidth=0.8, linestyle='--', label=f'+q/2 = +{qq/2:.4f}V')
plt.axvline(-qq/2, color='red',   linewidth=0.8, linestyle='--', label=f'-q/2 = -{qq/2:.4f}V')
plt.axhline(1/qq, color='darkorange',   linewidth=0.8, linestyle='--', label=f'1/q = {1/qq:.4f}V')
plt.grid(True)
plt.legend()
plt.title("Histograma del error de cuantización (sin ruido)")
plt.xlabel("Error [V]")
plt.ylabel("Densidad")

# AUTOCORRELACION
plt.subplot(1,2,2)

r = np.correlate(e, e, mode = 'full')
lags = np.arange(-len(e)+1, len(e))
L = 50 # Cantidad de lags que quiero ver

plt.plot(lags, r, label = 'Autocorrelacion', color = 'seagreen')
plt.xlim(-L, L)
plt.axvline(0, color='red',   linewidth=0.8, linestyle='--', label='Lag = 0')
plt.legend()
plt.grid(True)
plt.title("Autocorrelación del error (sin rudio)")
plt.xlabel("Retardo (lag)")
plt.ylabel("Autocorrelación")

#%% FFT
# Le pasamos secuencias reales, y me va a devolver secuencias complejas (modulo y fase)

XX = np.fft.fft(xx)

XXmod = np.abs(XX) # MODULO
XXfase = np.angle(XX) # FASE --> arcotangente de la parte real sobre la parte imaginaria --> va de -pi a pi --> hago un arcotangente de cuatro cuadrantes (lo comun es de dos cuadrantes)

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.plot(XXmod)
plt.title('Modulo de la DFT de xx')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(XXfase)
plt.title('Fase de la DFT de xx')
plt.grid(True)

plt.tight_layout()
plt.show()

#%% ESPECTRO DE LA SEÑAL Y EL RUIDO 

# Señal
XXmod = XXmod / len(XX)
XXmod_db = 20 * np.log10(XXmod)

# Ruido 
U_N = np.fft.fft(U_n)
UN_mod = np.abs(U_N)
UN_mod = np.abs(U_N) / len(U_N)
UN_mod_db = 20 * np.log10(UN_mod)

# Eje de frecuencias
freqs = np.fft.fftfreq(len(xx), d=ts)

# Como voy a graficar el modulo, solo quiero la primer mitad de la señal, ya que al ser par luego se repite
mitad = len(freqs) // 2
freqs = freqs[:mitad]
XXmod_db = XXmod_db[:mitad]
UN_mod_db = UN_mod_db[:mitad]

# Grafico
plt.figure()
plt.plot(freqs, XXmod_db, label = 'Señal pura')
plt.plot(freqs, UN_mod_db, label = 'Ruido')
plt.legend()
plt.title("Espectro de la señal original y el ruido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid()
plt.show()

# Yo defini SNR = 30, lo que significa que la potencia de la señal es 10^(30/10) = 1000 veces mayor que la potencia del ruido
# En terminos de amplitud, eso significa que el pico de la senoidal deberia estar unos 30dB por encima del piso de ruido

 

