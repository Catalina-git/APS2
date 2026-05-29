import numpy as np
import matplotlib.pyplot as plt

from scipy import signal as sig

# Frecuencia 
fs = 1000 # Hz

# Plantilla de diseño del filtro 
wp1 = 1 # Hz
wp2 = 35 # Hz --> 30 es aprox el ancho de banda que nos dio en la TS5
ws1 = .1 # Hz \
ws2 = 45 # Hz
gpass = 1 # dB
gstop = 40 # dB 

wp = [wp1, wp2] # Comienzo y fin de la banda de paso
ws = [ws1, ws2] # Comienzo y fin de la banda de stop

sos_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'butter', output = 'sos', fs = fs)
# Es de maxima planicidad en ambas bandas 

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cauer', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cauer)
# Es equiripple en la banda de paso y en la banda de stop 

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby1', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cheby 1) 
# Es equiripple en la banda de paso y max planicidad (parecido a max planicidad) en banda de stop 

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby2', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cheby 2) 
# Es equiripple en la banda de stop y max planicidad (parecido a max planicidad) en banda de paso

taps = sos_coeff.shape[0] * 2 # El por dos es porque son todos de segundo orden 

omega, resp_freq = sig.freqz_sos(sos_coeff, worN = 4096, fs = fs)

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

#%% Diagrama de polos y ceros 
# Paso SOS -> ZPK
z, p, k = sig.sos2zpk(sos_coeff)

fig, ax = plt.subplots(figsize=(6,6))

# Circunferencia unitaria
theta = np.linspace(0, 2*np.pi, 1000)

ax.plot(np.cos(theta), np.sin(theta), 'k--')

# Ceros
ax.plot(np.real(z),np.imag(z),'ob',markersize=10, label='Ceros')

# Polos
ax.plot(np.real(p), np.imag(p), 'xr', markersize=10, label='Polos')
ax.set_title('Diagrama de polos y ceros')
ax.set_xlabel('Parte real')
ax.set_ylabel('Parte imaginaria')
ax.grid(True)
ax.axis('equal')

ax.legend()

plt.show()

#%%
# RESPUESTA + PLANTILLA

fig, ax = plt.subplots(figsize=(12,6))

# Respuesta del filtro
ax.plot(omega,20*np.log10(np.abs(resp_freq)),lw=2,label='Respuesta del filtro (SOS)')


# ZONAS DE LA PLANTILLA
# Banda stop izquierda
ax.fill_between(omega,-60,-gstop,where=(omega <= ws1),color='red', alpha=0.15)

# Banda de paso
ax.fill_between(omega,-gpass,1,where=((omega >= wp1) & (omega <= wp2)), color='green',alpha=0.2,label='bw digital')

# Banda stop derecha
ax.fill_between(omega,-60,-gstop,where=(omega >= ws2),color='red',alpha=0.15)


# CONTORNO DE LA PLANTILLA
ax.plot([0, ws1, ws1, wp1, wp1, wp2, wp2, ws2, ws2, 100],[-gstop, -gstop, -gstop, -gstop, 0, 0, -gstop, -gstop, -gstop, -gstop],'k--',lw=1.5,label='plantilla')


# ESTÉTICA
ax.set_title('Filtro Pasa-Banda IIR (SOS) - Respuesta y plantilla',fontsize=18)

ax.set_xlabel('Frecuencia [Hz]', fontsize=14)
ax.set_ylabel('Modulo [dB]', fontsize=14)

ax.set_xlim([0,100])
ax.set_ylim([-60,5])

ax.grid(True, alpha=0.3)

ax.legend(fontsize=13)

plt.show()


