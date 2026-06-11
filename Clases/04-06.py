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
ws1 = .1 # Hz 
ws2 = 45 # Hz
gpass = 1 # dB
gstop = 40 # dB 

wp = (wp1, wp2) # Comienzo y fin de la banda de paso
ws = (ws1, ws2) # Comienzo y fin de la banda de stop


# HACK para la plantilla de diseño
wp1_hack = .4 # Hz
wp2_hack = 35 # Hz --> 30 es aprox el ancho de banda que nos dio en la TS5
ws1_hack = .3 # Hz 
ws2_hack = 35.4 # Hz
gpass_hack = 1 # dB
gstop_hack = 40 # dB 

wp_hack = (wp1_hack, wp2_hack) # Comienzo y fin de la banda de paso
ws_hack = (ws1_hack, ws2_hack) # Comienzo y fin de la banda de stop

#%% FIRWIN2
# Me devuelve los coeficientes 'b' (los coeficientes 'a' son cero menos el 'a0' que vale 1)
# Como no hay coeficientes 'ai', no hay recurcion, no hay problemas numericos exagerados --> puedo implementar sos

numtaps = 2000
# Si tengo 8000 taps --> en tiempo tengo: N = 8000 y la distancia entre muestras es N*Ts = N/fs = 8segundos

freq = np.array([0, ws1_hack, wp1_hack, wp2_hack, ws2_hack, fs//2]) # Tinee que ir siempre para arriba
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
plot_plantilla('bandpass', wp, gpass, ws, gstop, fs)
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

#%% POLOS Y CEROS
# fig, ax = plt.subplots(figsize=(6,6))

# # Circunferencia unitaria
# theta = np.linspace(0, 2*np.pi, 1000)

# # Ceros
# ax.plot(np.cos(theta), np.sin(theta), 'k--')
# ax.plot(np.real(ceros),np.imag(ceros),'o',markersize = 10, label='Ceros')

# # Polos
# ax.plot(np.real(polos), np.imag(polos), 'x', markersize = 10, label='Polos')
# ax.set_title('Diagrama de polos y ceros')
# ax.set_xlabel('Parte real')
# ax.set_ylabel('Parte imaginaria')
# ax.grid(True)
# ax.axis('equal')

# ax.legend()
# plt.show()


#%% Procesar el ECG
# import scipy.io as sio

# fs_ecg = 1000 # Hz

# # ECG con ruido 
# mat_struct = sio.loadmat('./ECG_TP4.mat')

# ecg_one_lead = mat_struct['ecg_lead']. flatten()

# sos_f_butter = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'butter', output = 'sos', fs = fs)
# sos_f_cauer = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cauer', output = 'sos', fs = fs)
# sos_f_cheby2 = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby2', output = 'sos', fs = fs)

# ECG_f_butter = sig.sosfiltfilt(sos_f_butter, ecg_one_lead)
# ECG_f_cauer = sig.sosfiltfilt(sos_f_cauer, ecg_one_lead)
# ECG_f_cheby2 = sig.sosfiltfilt(sos_f_cheby2, ecg_one_lead)

# ECG_f_win = sig.filtfilt(fir_win, 1, ecg_one_lead)

# %% GRAFICOS — ZONAS SIN RUIDO

# regiones_sin_ruido = [
#     [4000, 5500],
#     [10000, 11000]
# ]

# for r in regiones_sin_ruido:
#     a,b = r
#     t = np.arange(a,b)

#     plt.figure(figsize=(10,4))
#     plt.plot(t, ecg_one_lead[t], label="ECG")
#     plt.plot(t, ECG_f_butter[t], label="Butterworth")
#     plt.plot(t, ECG_f_win[t], label="FIR")
#     plt.title('ECG sin ruido desde ' + str(r[0]) + ' hasta ' + str(r[1]) )
#     plt.ylabel('Adimensional')
#     plt.xlabel('Muestras (#)')
#     plt.legend()
#     plt.grid()

# %% GRAFICOS — ZONAS CON RUIDO

# regiones_ruidosas = [
#     (np.array([5, 5.2])*60*fs).astype(int),
#     (np.array([12, 12.4])*60*fs).astype(int),
#     (np.array([15, 15.2])*60*fs).astype(int)
# ]

# for r in regiones_ruidosas:
#     a,b = r
#     t = np.arange(a,b)

#     plt.figure(figsize=(10,4))
#     plt.plot(t, ecg_one_lead[t], label="ECG")
#     plt.plot(t, ECG_f_butter[t], label="Butterworth")
#     plt.plot(t, ECG_f_win[t], label="FIR")
#     plt.title('ECG con ruido desde ' + str(r[0]) + ' hasta ' + str(r[1]) )
#     plt.ylabel('Adimensional')
#     plt.xlabel('Muestras (#)')
#     plt.legend()
#     plt.grid()

#%% FIR con CUADRADOS MINIMOS 
numtaps = 2001
# Si tengo 8000 taps --> en tiempo tengo: N = 8000 y la distancia entre muestras es N*Ts = N/fs = 8segundos

freq = np.array([0, ws1_hack, wp1_hack, wp2_hack, ws2_hack, fs//2]) # Tinee que ir siempre para arriba
# El fs/2 es Nyquist y es porque no esta normalizado

# Ganancia, banda y peso 
band = [0, ws1_hack, wp1_hack, wp2_hack, ws2_hack, fs/2]
gain  = [0, 0, 1, 1, 0, 0]
weight = [10, 1, 5]

# FIR con cuadrados minimos
fir_ls = sig.firls(numtaps, band, gain, weight = weight, fs = fs)

# Respuesta en frecuencia
ls_fir, h_fir_ls = sig.freqz(fir_ls, worN = ww, fs = fs)
fase_fir_ls = np.unwrap(np.angle(h_fir_ls))
gd_fir_ls = -np.diff(fase_fir_ls) / np.diff(ls_fir/fs*np.pi)

# GRAFICOS
# MODULO
plt.figure(figsize=(12,10))

plt.plot(ls_fir, 20*np.log10(abs(h_fir_ls)), label="FIR")
plot_plantilla('bandpass', wp, gpass, ws, gstop, fs)
plt.title('FIR con cuadrados minimos - Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(ω)| [dB]')
#plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

plt.show()

# FIR ANTISIMETRICO

# Eje de simetria
numtaps = 1835
M = numtaps - 1

# FIR con cuadrados minimos
# Vuelvo a llamar a la funcion porque cambie los numtaps
# fir_ls = sig.firls(numtaps, band, gain, weight = weight, fs = fs)

# Copia para modificar
fir_antisym = fir_ls.copy() 

# Forzar coeficiente central a cero si numtaps es impar
if numtaps % 2 == 1:
    fir_antisym[M//2] = 0.0

# De la mitad en adelante le cambio el signo 
fir_antisym[M:numtaps] = fir_antisym[M:numtaps] * (-1)

# Respuesta en frecuencia FIR antisimetrico 
ls_fir_ant, h_fir_ls_ant = sig.freqz(fir_antisym, worN = ww, fs = fs)
fase_fir_ls_ant = np.unwrap(np.angle(h_fir_ls_ant))
gd_fir_ls_ant = -np.diff(fase_fir_ls_ant) / np.diff(ls_fir_ant/fs*np.pi)

# GRAFICOS
# MODULO
plt.figure(figsize=(12,10))

plt.plot(ls_fir_ant, 20*np.log10(abs(h_fir_ls_ant)), label="FIR")
plot_plantilla('bandpass', wp, gpass, ws, gstop, fs)
plt.title('FIR con cuadrados minimos - Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(ω)| [dB]')
#plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

plt.show()



