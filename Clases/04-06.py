import numpy as np
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
from scipy import signal as sig

# Diseño de filtros FIR 

# firwin o firwin2 --> usa ventana
# La uno solo me deja hacer tipo 1 y 2, la firwin2 me deja usar todos 
# firls --> cuadrados minimos
# remez --> diseño de filtros en base al metodo de park Mc --> optimizacion
# Los ultimos dos son "mejores", atenuan mas rapido

# Frecuencia 
fs = 1000 # Hz

# ww es para el WorN
ww = np.concatenate([np.logspace(start = -2, stop = 0.1, num = 500),
                np.linspace(start = 1.26, stop = 35, num = 200),
                np.logspace(start = 1.55, stop = 1.65, num = 300),
                np.linspace(start = 46, stop = fs//2, num = 50)])
ww = np.sort(ww)

# Plantilla de diseño del filtro 
wp1 = 1 # Hz
wp2 = 35 # Hz --> 30 es aprox el ancho de banda que nos dio en la TS5
ws1 = .1 # Hz \
ws2 = 45 # Hz
gpass = 1 # dB
gstop = 40 # dB 

wp = (wp1, wp2) # Comienzo y fin de la banda de paso
ws = (ws1, ws2) # Comienzo y fin de la banda de stop

#%% FIRWIN2
# Me devuelve los coeficientes 'b' (los coeficientes 'a' son cero menos el 'a0' que vale 1)
# Como no hay coeficientes 'ai', no hay recurcion, no hay problemas numericos exagerados --> puedo implementar sos

numtaps = 3600

freq = np.array([0, ws1, wp1, wp2, ws2, fs//2]) # Tinee que ir siempre para arriba
# El fs/2 es Nyquist y es porque no esta normalizado

# gain = 10**(((-1)*np.array([gstop, gstop, gpass, gpass, gstop, gstop])/20)) # Es la respuesta deseada
gain = np.array([0, 0, 1, 1, 0, 0])

# if numtaps % 2 == 0: # '%' es el resto de la division
#     gain[-1] = 0 # Fuerzo para que termine en cero, que en Nyquist haya un cero --> esto es porque al tener un filtro de tipo II (numtaps 30), tengo un cero topologico en Nyquist
    

# FIR con ventana rectangular
fir_win = sig.firwin2(numtaps, freq, gain, nfreqs = 2**14, window = 'boxcar', fs = fs)
retardo = (numtaps - 1)//2

plt.figure()
plt.title('Filtro FIR por metodo de ventanas')
plt.plot(fir_win)
plt.grid()
plt.show()

# Respuesta en frecuencia
w_fir, h_fir = sig.freqz(fir_win, worN = ww, fs = fs)
fase_fir = np.unwrap(np.angle(h_fir))
gd_fir = -np.diff(fase_fir) / np.diff(w_fir/fs*np.pi)

ceros, polos, k = sig.tf2zpk(fir_win, a = 1)

# GRAFICOS

# MODULO
plt.figure(figsize=(12,10))

plt.subplot(3,1,1)
plt.plot(w_fir, 20*np.log10(abs(h_fir)), label="FIR")
plot_plantilla('bandpass', wp, gpass*2, ws, gstop*2, fs)
plt.title('FIR ventana rectangular - Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(ω)| [dB]')
#plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

# FASE
plt.subplot(3,1,2)
plt.plot(w_fir, fase_fir)
plt.title('FIR ventana rectangular - Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
# plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

# RETARDO
plt.subplot(3,1,3)
plt.plot(w_fir[1:], gd_fir)
plt.title('FIR ventana rectangular - Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
#plt.xlim(0, 65)
plt.grid(True, which='both', ls=':')
plt.legend()

plt.tight_layout()
plt.show()

# POLOS Y CEROS
fig, ax = plt.subplots(figsize=(6,6))

# Circunferencia unitaria
theta = np.linspace(0, 2*np.pi, 1000)

# Ceros
ax.plot(np.cos(theta), np.sin(theta), 'k--')
ax.plot(np.real(ceros),np.imag(ceros),'o',markersize = 10, label='Ceros')

# Polos
ax.plot(np.real(polos), np.imag(polos), 'x', markersize = 10, label='Polos')
ax.set_title('Diagrama de polos y ceros')
ax.set_xlabel('Parte real')
ax.set_ylabel('Parte imaginaria')
ax.grid(True)
ax.axis('equal')

ax.legend()
plt.show()


#%% Procesar el ECG
import scipy.io as sio

fs_ecg = 1000 # Hz

# ECG con ruido 
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']. flatten()

sos_f_butter = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'butter', output = 'sos', fs = fs)
sos_f_cauer = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cauer', output = 'sos', fs = fs)
sos_f_cheby2 = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby2', output = 'sos', fs = fs)

ECG_f_butter = sig.sosfiltfilt(sos_f_butter, ecg_one_lead)
ECG_f_cauer = sig.sosfiltfilt(sos_f_cauer, ecg_one_lead)
ECG_f_cheby2 = sig.sosfiltfilt(sos_f_cheby2, ecg_one_lead)

ECG_f_win = sig.filtfilt(fir_win, 1, ecg_one_lead)

# %% GRAFICOS — ZONAS SIN RUIDO

regiones_sin_ruido = [
    [4000, 5500],
    [10000, 11000]
]

for r in regiones_sin_ruido:
    a,b = r
    t = np.arange(a,b)

    plt.figure(figsize=(10,4))
    plt.plot(t, ecg_one_lead[t], label="ECG")
    plt.plot(t, ECG_f_butter[t], label="Butterworth")
    plt.plot(t, ECG_f_win[t], label="FIR")
    plt.title('ECG sin ruido desde ' + str(r[0]) + ' hasta ' + str(r[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.legend()
    plt.grid()

# %% GRAFICOS — ZONAS CON RUIDO

regiones_ruidosas = [
    (np.array([5, 5.2])*60*fs).astype(int),
    (np.array([12, 12.4])*60*fs).astype(int),
    (np.array([15, 15.2])*60*fs).astype(int)
]

for r in regiones_ruidosas:
    a,b = r
    t = np.arange(a,b)

    plt.figure(figsize=(10,4))
    plt.plot(t, ecg_one_lead[t], label="ECG")
    plt.plot(t, ECG_f_butter[t], label="Butterworth")
    plt.plot(t, ECG_f_win[t], label="FIR")
    plt.title('ECG con ruido desde ' + str(r[0]) + ' hasta ' + str(r[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.legend()
    plt.grid()
