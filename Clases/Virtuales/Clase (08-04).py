#%% LLamo a las librerias que se van a utilizar
import numpy as np
import matplotlib.pyplot as plt

#%% Item a
# Defino los parámetros a utilizar
n = 1000           # muestras
fs = 1000          # Hz
# f0 = fs / n        # Hz
f0 = n / 4
ts = 1/fs
tt = np.arange(n) * ts

# Señal (potencia unitaria)
vmax = np.sqrt(2)
dc = 0
xx = dc + vmax * np.sin(2*np.pi*f0*tt)

#%% ADC 
B = 4 # 4 bits --> 16 niveles
VF = 2  # Rango +-2V
q = (2*VF) / (2**B)   # Paso de cuantización
Pq = (q**2) / 12      # Potencia de cuantización

# Defino el ruido
kn = 10
Pn = kn * Pq
mu = 0
U_n = np.random.normal(mu, np.sqrt(Pn), n)

# Señal con ruido
sr = xx + U_n

# Cuantización (ADC)
sq = np.round(sr / q) * q

#%% FFT

N = len(xx)

# Señal pura
XX = np.fft.fft(xx)
XXmod = np.abs(XX) / N
XXmod_db = 20 * np.log10(XXmod + 1e-12) # El +1e-12 es una medida para evitar el warning del logaritmo, para que no de cero
# Es un valor que no afecta al calculo computacional que estoy haciendo

# Señal con ruido (entrada ADC)
SR = np.fft.fft(sr)
SRmod = np.abs(SR) / N
SRmod_db = 20 * np.log10(SRmod + 1e-12)

# Señal cuantizada (salida ADC)
SQ = np.fft.fft(sq)
SQmod = np.abs(SQ) / N
SQmod_db = 20 * np.log10(SQmod + 1e-12)

# Eje de frecuencias
freqs = np.fft.fftfreq(N, d=ts)

# Me quedo con la mitad positiva (eso es porque grafico el modulo)
mitad = N // 2
freqs = freqs[:mitad]

XXmod_db = XXmod_db[:mitad]
SRmod_db = SRmod_db[:mitad]
SQmod_db = SQmod_db[:mitad]

# Pisos teóricos
piso_analog = 10 * np.log10(Pn)
piso_digital = 10 * np.log10(Pq)

#%% Error de cuantización
e = sq - sr

#%% Graficos

# Entrada y salida del ADC
plt.figure()

plt.plot(tt, sq, label=r'$s_Q = Q_B(s_R)$ (ADC out)')
plt.plot(tt, sr, '.', markersize=2, label=r'$s_R = s + n$ (ADC in)')
plt.plot(tt, xx, '--', label='s (analog)')

plt.title(f'Señal muestreada por un ADC de {B} bits ±VF = {VF} V - q = {q:.3f} V')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')

plt.grid()
plt.legend()
plt.show()

# Espectro de la FFT
plt.figure()

plt.plot(freqs, SQmod_db, color = 'aquamarine', label=r'$s_Q = Q_B(s_R)$ (ADC out)')
# plt.plot(freqs, XXmod_db, color = 'forestgreen', label='s (analog)')
plt.plot(freqs, SRmod_db, ':', color = 'darkorange', label=r'$s_R = s + n$ (ADC in)')

plt.axhline(piso_analog, linestyle='--', color = 'red', label=f'{piso_analog:.2f} dB (piso analógico)')
plt.axhline(piso_digital, linestyle='--', color = 'dodgerblue', label=f'{piso_digital:.2f} dB (piso digital)')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de potencia [dB]')
plt.title(f'Señal muestreada por un ADC de {B} bits - ±VF = {VF} V - q = {q:.3f} V')

plt.grid()
plt.legend()
plt.show()


# Histograma del error de cuantizacion
plt.figure()

plt.hist(e, bins=20, density=True)

plt.axvline(q/2, linestyle='--', color='orange', label='+q/2')
plt.axvline(-q/2, linestyle='--', color = 'red', label='-q/2')
plt.axhline(1/q, linestyle='--', color = 'orchid', label='1/q')

plt.title(f'Ruido de cuantización para {B} bits - ±VF = {VF} V - q = {q:.3f} V')
plt.xlabel('Error [V]')
plt.ylabel('Densidad')

plt.grid()
plt.legend()
plt.show()

#%% 
# El nivel medio del piso de ruido deberia estar en 10*log(Pn + Pq) 
# --> donde Pn + Pq = Pr rayita = Potencia total de ruido de la entrada del ADC 
# En nuestro caso, deberia estar en aproximadamente -45,64 dB --> se verifica con el grafico

# Cuenta para ver si me da donde deberia el piso de ruido
piso_ruido = 10 * np.log10(Pn + Pq)

# piso_ruido = 10 * np.log10((Pn + Pq) / (N/2))