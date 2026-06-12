import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from pytc2.sistemas_lineales import plot_plantilla

# Frecuencia 
fs = 1000 # Hz

# ww es para el WorN
ww = np.concat([np.logspace(start = -2, stop = 0.1, num = 500),
                np.linspace(start = 1.26, stop = 35, num = 200),
                np.logspace(start = 1.55, stop = 1.65, num = 300),
                np.linspace(start = 46, stop = fs//2, num = 50)])

# Plantilla de diseño del filtro 
wp1 = 1 # Hz
wp2 = 35 # Hz --> 30 es aprox el ancho de banda que nos dio en la TS5
ws1 = .1 # Hz \
ws2 = 45 # Hz
gpass = 1 # dB
gstop = 40 # dB 

wp = (wp1, wp2) # Comienzo y fin de la banda de paso
ws = (ws1, ws2) # Comienzo y fin de la banda de stop

# Funciones de aproximacion 
# ftype = 'butter'
# ftype = 'cheby1'
# ftype = 'cheby2'
# ftype = 'cauer'

#%% Funciones de aproximacion 
sos_f_butter = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'butter', output = 'sos', fs = fs) # Es de maxima planicidad en ambas bandas 
sos_f_cauer = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cauer', output = 'sos', fs = fs)
sos_f_cheby2 = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby2', output = 'sos', fs = fs)

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cauer', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cauer)
# Es equiripple en la banda de paso y en la banda de stop 

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby1', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cheby 1) 
# Es equiripple en la banda de paso y max planicidad (parecido a max planicidad) en banda de stop 

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby2', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cheby 2) 
# Es equiripple en la banda de stop y max planicidad (parecido a max planicidad) en banda de paso

#%% BUTTERWORTH
taps_butter = sos_f_butter.shape[0] * 2 # El por dos es porque son todos de segundo orden 
omega_butter, resp_freq_butter = sig.freqz_sos(sos_f_butter, worN = ww, fs = fs)
phase_butter = np.unwrap(np.angle(resp_freq_butter))
gd_butter = -np.diff(phase_butter) / np.diff(ww)
gd_butter = np.append(gd_butter[0], gd_butter)

# GRAFICOS

# MODULO
plt.figure(figsize=(12,10))

plt.subplot(3,1,1)
plt.plot(omega_butter, 20*np.log10(abs(resp_freq_butter)), label="FIR")
plot_plantilla('bandpass', wp, gpass, ws, gstop, fs)
plt.title('IIR Butterworth - Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(ω)| [dB]')
# plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

# FASE
plt.subplot(3,1,2)
plt.plot(omega_butter, phase_butter)
plt.title('IIR Butterworth - Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.xlim(0, 35)
plt.grid(True, which='both', ls=':')
plt.legend()

# RETARDO
plt.subplot(3,1,3)
plt.plot(ww, gd_butter)
plt.title('IIR Butterworth - Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.xlim(0, 35)
plt.grid(True, which='both', ls=':')
plt.legend()

plt.tight_layout()
plt.show()

#%% CAUER O ELIPTICO 
taps_cauer = sos_f_cauer.shape[0] * 2 # El por dos es porque son todos de segundo orden 
omega_cauer, resp_freq_cauer = sig.freqz_sos(sos_f_cauer, worN = ww, fs = fs)
phase_cauer = np.unwrap(np.angle(resp_freq_cauer))
gd_cauer = -np.diff(phase_cauer) / np.diff(ww)
gd_cauer = np.append(gd_cauer[0], gd_cauer)

# GRAFICOS

# MODULO
plt.figure(figsize=(12,10))

plt.subplot(3,1,1)
plt.plot(omega_cauer, 20*np.log10(abs(resp_freq_cauer)), label="FIR")
plot_plantilla('bandpass', wp, gpass, ws, gstop, fs)
plt.title('IIR Cauer o Elíptico - Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(ω)| [dB]')
# plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

# FASE
plt.subplot(3,1,2)
plt.plot(omega_cauer, phase_cauer)
plt.title('IIR Cauer o Elíptico - Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.xlim(0, 35)
plt.grid(True, which='both', ls=':')
plt.legend()

# RETARDO
plt.subplot(3,1,3)
plt.plot(ww, gd_cauer)
plt.title('IIR Cauer o Elíptico - Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.xlim(0, 35)
plt.grid(True, which='both', ls=':')
plt.legend()

plt.tight_layout()
plt.show()

#%% CHEBYSHEV TIPO 2
taps_cheby2 = sos_f_cheby2.shape[0] * 2 # El por dos es porque son todos de segundo orden 
omega_cheby2, resp_freq_cheby2 = sig.freqz_sos(sos_f_cheby2, worN = ww, fs = fs)
phase_cheby2 = np.unwrap(np.angle(resp_freq_cheby2))
gd_cheby2 = -np.diff(phase_cheby2) / np.diff(ww)
gd_cheby2 = np.append(gd_cheby2[0], gd_cheby2)

# GRAFICOS

# MODULO
plt.figure(figsize=(12,10))

plt.subplot(3,1,1)
plt.plot(omega_cheby2, 20*np.log10(abs(resp_freq_cheby2)), label="FIR")
plot_plantilla('bandpass', wp, gpass, ws, gstop, fs)
plt.title('IIR Cauer o Elíptico - Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(ω)| [dB]')
# plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

# FASE
plt.subplot(3,1,2)
plt.plot(omega_cheby2, phase_cheby2)
plt.title('IIR Cauer o Elíptico - Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.xlim(0, 35)
plt.grid(True, which='both', ls=':')
plt.legend()

# RETARDO
plt.subplot(3,1,3)
plt.plot(ww, gd_cheby2)
plt.title('IIR Cauer o Elíptico - Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.xlim(0, 35)
plt.grid(True, which='both', ls=':')
plt.legend()

plt.tight_layout()
plt.show()

#%% Como implementamos el filtro? 
# Para usarlo necesitamos la señal 
import scipy.io as sio

fs_ecg = 1000 # Hz

# ECG con ruido 
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']. flatten()

# N = len(ecg_one_lead) 

# Limito la energia al ancho de banda para ver bien la señal electrocardiografica 
# Usamos la funcion de scipy.signal --> sosfilt --> va a tener limitaciones 

ECG_f_butter = sig.sosfilt(sos_f_butter, ecg_one_lead)
ECG_f_cauer = sig.sosfilt(sos_f_cauer, ecg_one_lead)
ECG_f_cheby2 = sig.sosfilt(sos_f_cheby2, ecg_one_lead)

plt.figure()
plt.plot(ecg_one_lead, label = 'ECG original con ruido')
plt.plot(ECG_f_butter, label = 'ECG filtrado (sosfilt) - Butter')
plt.plot(ECG_f_cauer, label = 'ECG filtrado (sosfilt) - Cauer')
plt.plot(ECG_f_cheby2, label = 'ECG filtrado (sosfilt) - Cheby 2')
plt.legend()
plt.show()

#%% Vamos a usar sosfiltfilt
# Hace un backward - fordward 
# Es un filtrado bidireccional, y lo que asume es como que pasas dos veces por la respuesta de modulo 
# Es decir, voy a tener 2alfa max de ripple, y 2alfa min de atenuacion 
# Y en dB es como tener el doble de dB
# El efecto de invertir en tiempo es util para el retardo de fase? Neutraliza la fase, sincroniza temporalmente la señal original con la filtrada

sos_ff_butter = sig.iirdesign(wp, ws, gpass / 2, gstop / 2, analog = False, ftype = 'butter', output = 'sos', fs = fs)
sos_ff_cauer = sig.iirdesign(wp, ws, gpass / 2, gstop / 2, analog = False, ftype = 'cauer', output = 'sos', fs = fs)
sos_ff_cheby2 = sig.iirdesign(wp, ws, gpass / 2, gstop / 2, analog = False, ftype = 'cheby2', output = 'sos', fs = fs)

taps = sos_ff_butter.shape[0] * 2 # El por dos es porque son todos de segundo orden 
omega, resp_freq = sig.freqz_sos(sos_ff_butter, worN = ww, fs = fs)
phase = np.unwrap(np.angle(resp_freq))

ECG_ff_butter = sig.sosfiltfilt(sos_ff_butter, ecg_one_lead)
ECG_ff_cauer = sig.sosfiltfilt(sos_ff_cauer, ecg_one_lead)
ECG_ff_cheby2 = sig.sosfiltfilt(sos_ff_cheby2, ecg_one_lead)

plt.figure()
plt.plot(ecg_one_lead, label = 'ECG original con ruido')
plt.plot(ECG_ff_butter, label = 'ECG filtrado (sosfiltfilt) - Butter')
plt.plot(ECG_ff_cauer, label = 'ECG filtrado (sosfiltfilt) - Cauer')
plt.plot(ECG_ff_cheby2, label = 'ECG filtrado (sosfiltfilt) - Cheby 2')
plt.legend()
plt.show()

# Probar con todas las funciones de aproximacion 

#%% Regiones de interés con ruido
 
N = len(ecg_one_lead) 
cant_muestras = N

regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ECG_f_butter[zoom_region], label='Butter', linewidth=2)
    plt.plot(zoom_region, ECG_f_cauer[zoom_region], label='Cauer', linewidth=2)
    plt.plot(zoom_region, ECG_f_cheby2[zoom_region], label='Cheby 2', linewidth=2)
   
    plt.title('ECG with noise filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()

#%% Regiones de interés sin ruido
 
regs_interes = (
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ECG_ff_butter[zoom_region], label='Butter', linewidth=2)
    plt.plot(zoom_region, ECG_ff_cauer[zoom_region], label='Cauer', linewidth=2)
    plt.plot(zoom_region, ECG_ff_cheby2[zoom_region], label='Cheby 2', linewidth=2)
   
    plt.title('ECG without noise filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()

