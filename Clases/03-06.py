import numpy as np
import matplotlib.pyplot as plt

from scipy import signal as sig

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

wp = [wp1, wp2] # Comienzo y fin de la banda de paso
ws = [ws1, ws2] # Comienzo y fin de la banda de stop

# Funciones de aproximacion 
# ftype = 'butter'
# ftype = 'cheby1'
# ftype = 'cheby2'
# ftype = 'cauer'

sos_f_butter = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'butter', output = 'sos', fs = fs)
# Es de maxima planicidad en ambas bandas 
sos_f_cauer = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cauer', output = 'sos', fs = fs)
sos_f_cheby2 = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby2', output = 'sos', fs = fs)

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cauer', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cauer)
# Es equiripple en la banda de paso y en la banda de stop 

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby1', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cheby 1) 
# Es equiripple en la banda de paso y max planicidad (parecido a max planicidad) en banda de stop 

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby2', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cheby 2) 
# Es equiripple en la banda de stop y max planicidad (parecido a max planicidad) en banda de paso

taps = sos_f_butter.shape[0] * 2 # El por dos es porque son todos de segundo orden 

omega, resp_freq = sig.freqz_sos(sos_f_butter, worN = ww, fs = fs)

phase = np.unwrap(np.angle(resp_freq))

# Graficos 
fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, tight_layout = True)
ax1, ax2 = axs

# Modulo
ax1.set_title(f"Filtro Pasa-Banda IIR (SOS) + Plantilla de diseño - {taps} tap") # taps es la cantidad de coeficientes --> me va a dar el orden del filtro 
ax1.plot(omega, 20 * np.log10(abs(resp_freq)), 'C0')
ax1.set_ylabel("Amplitude in dB", color = 'C0')
ax1.set_ylabel('Magnitud [dB]')
ax1.grid(True)
ax1.set_xlim([0,100])
ax1.set_ylim([-60,5])
ax1.legend()

# PLANTILLA DE DISEÑO
# Banda stop izquierda
ax1.fill_between(omega,-60,-gstop,where=(omega <= ws1),color='red', alpha=0.15)

# Banda de paso
ax1.fill_between(omega,-gpass,1,where=((omega >= wp1) & (omega <= wp2)), color='green',alpha=0.2,label='bw digital')

# Banda stop derecha
ax1.fill_between(omega,-60,-gstop,where=(omega >= ws2),color='red',alpha=0.15)

# CONTORNO DE LA PLANTILLA
ax1.plot([0, ws1, ws1, wp1, wp1, wp2, wp2, ws2, ws2, 100],[-gstop, -gstop, -gstop, -gstop, 0, 0, -gstop, -gstop, -gstop, -gstop],'k--',lw=1.5,label='plantilla')

# Fase
phase = np.unwrap(np.angle(resp_freq)) # En este caso si me conviene usar unwrap, porque evito la discontinuidad en la fase
ax2.plot(omega, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color = 'C1')
ax2.grid(True)
ax2.axis('tight')

plt.show()

#%% RETARDO DE GRUPO

gd = -np.diff(phase) / np.diff(ww)
gd = np.append(gd[0], gd)

fig, axs = plt.subplots(figsize=(6,6))

axs.set_title('Retardo de grupo') 
axs.plot(ww, gd, 'm')
axs.set_xlabel('Frecuencia [Hz]')
axs.set_ylabel('Retardo [muetras]')
axs.grid(True, alpha = 0.5)
axs.set_xlim(-10,fs/2)
# axs.set_ylim([-20,20])
axs.legend()

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

plt.figure()
plt.plot(ecg_one_lead, label = 'ECG original con ruido')
plt.plot(ECG_f_butter, label = 'ECG filtrado (sosfilt)')
plt.legend()
plt.show()

# Probar espectro de entrada vs espectro de salida? 

#%% Vamos a usar sosfiltfilt
# Hace un backward - fordward 
# Es un filtrado bidireccional, y lo que asume es como que pasas dos veces por la respuesta de modulo 
# Es decir, voy a tener 2alfa max de ripple, y 2alfa min de atenuacion 
# Y en dB es como tener el doble de dB
# El efecto de invertir en tiempo es util para el retardo de fase? Neutraliza la fase, sincroniza temporalmente la señal original con la filtrada

sos_ff_butter = sig.iirdesign(wp, ws, gpass / 2, gstop / 2, analog = False, ftype = 'butter', output = 'sos', fs = fs)

taps = sos_ff_butter.shape[0] * 2 # El por dos es porque son todos de segundo orden 

omega, resp_freq = sig.freqz_sos(sos_ff_butter, worN = ww, fs = fs)

phase = np.unwrap(np.angle(resp_freq))

ECG_ff_butter = sig.sosfiltfilt(sos_ff_butter, ecg_one_lead)

plt.figure()
plt.plot(ecg_one_lead, label = 'ECG original con ruido')
plt.plot(ECG_ff_butter, label = 'ECG filtrado (sosfiltfilt)')
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
   
    plt.figure(1)
    plt.plot(zoom_region, ECG_f_butter[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
    #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
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
   
    plt.figure(2)
    plt.plot(zoom_region, ECG_ff_butter[zoom_region], label='ECG', linewidth=2)
    # plt.plot(zoom_region, yy_2[zoom_region], label='Butterworth')
    # plt.plot(zoom_region, yy_2[zoom_region + gd], label='FIR Window')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()

