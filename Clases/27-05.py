import numpy as np
import matplotlib.pyplot as plt

from scipy import signal as sig


#%% Ejemplo de guiado IIR

# Uso los coeficientes del ejemplo que hicimos en clase 
# T(z) = (z^2 + 2z + 1) / (z^2 + 0z + 0,95^2)
b_coeff = np.array([1, 2, 1])
a_coeff = np.array([1, 0, 0.95**2]) # Si disminuyo el ultimo coeficiente, bajo el Q, el grafico del modulo es cada vez mas lineal, y la respuesta de fase queda casi lineal 

taps = b_coeff.shape[0]

omega, resp_freq = sig.freqz(b_coeff, a = a_coeff, worN = 1024)

fig, ax1 = plt.subplots(tight_layout = True)
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter") # taps es la cantidad de coeficientes --> me va a dar el orden del filtro 
# ax1.plot(omega, 20 * np.log10(abs(resp_freq)), 'C0')
ax1.plot(omega, abs(resp_freq), 'C0')
ax1.set_ylabel("Amplitude in dB", color='C0')
ax1.set(xlabel="Frequency in rad/sample", xlim=(0, np.pi))

ax2 = ax1.twinx()

# phase = np.unwrap(np.angle(resp_freq))
phase = np.angle(resp_freq)

ax2.plot(omega, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color='C1')
ax2.grid(True)
ax2.axis('tight')
plt.show()


# El whole lo dejo en False para que me recorra de cero a Nyquist (fs/2)
# El linspace va a generar el eje de frecuencias ff --> tengo que pasarle en que frecuencias quiero que evalue, van a ser valores arbitrarios 
# freqz es respuesta de modulo y fase para el plano z

# iirdesign: wp = wc, ws = ws, gpass = alfamax, gstop = alfamin
# Los dos primeros w (wc y ws) son los vertices del filtro
# analog --> True = T(s), False = T(z)
# Lo que sigue son todas formas de escribir la T(s)
# SOS: productoria de los Pi(s)/Qi(s) --> en este caso es coeficiente de segundo orden, en la practica es una matriz --> preferimos usar estas 
# ba: P(s)/Q(s) --> dos vectores (polos y ceros)
#zpk: k*[((s-z0)*(s-z1)*...)/((s-p0)*(s-p1)*...)] --> dos vectores (polos y ceros) y un numero real (k)

#%% Vamos a armar la plantilla, y a partir de eso buscamos la respuesta en frecuencia 
# Filtro de orden n = 4
# alfamin = 10 dB
# alfamax = 3 dB
# ws = 100 Hz
# wc = 70 Hz 

# FILTRO ANALOGICO 
# Normalizo por wc
wp_n = 1 # es wc
ws_n = 10/7 # es ws
gpass = 3 # es alfamax
gstop = 10 # es alfamin

z_a,p_a,k_a = sig.iirdesign(wp_n, ws_n, gpass, gstop, analog = True, ftype = 'butter', output = 'zpk') # Hago primero el filtro analogico

# FILTRO DIGITAL
# Vuelvo a mis valores no normalizados y le paso a la funcion la frecuencia fs --> le paso la plantilla desnormalizada  
fs = 500 # Hz
wp = 70 # Hz --> es wc
ws = 100 # Hz --> es ws
gpass = 3 # dB --> es alfamax
gstop = 10 # dB --> es alfamin

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'butter', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Butter)

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cauer', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cauer)
# Es equiripple en la banda de paso y en la banda de stop 

b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby1', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cheby 1) 
# Es equiripple en la banda de paso y max planicidad (parecido a max planicidad) en banda de stop 

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby2', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cheby 2) 
# Es equiripple en la banda de stop y max planicidad (parecido a max planicidad) en banda de paso

taps = b_coeff.shape[0]

omega, resp_freq = sig.freqz(b_coeff, a = a_coeff, worN = 1024, fs = fs)

fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, tight_layout = True)
ax1, ax2 = axs
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter") # taps es la cantidad de coeficientes --> me va a dar el orden del filtro 
ax1.plot(omega, 20 * np.log10(abs(resp_freq)), 'C0')
# ax1.plot(omega, abs(resp_freq), 'C0')
ax1.set_ylabel("Amplitude in dB", color = 'C0')
ax1.grid(True)
ax1.axis('tight')
ax1.set(xlabel="Frequency in rad/sample", xlim = (0, np.pi))

# ax2 = ax1.twinx()

phase = np.unwrap(np.angle(resp_freq)) # En este caso si me conviene usar unwrap, porque evito la discontinuidad en la fase
# phase = np.angle(resp_freq)

ax2.plot(omega, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color = 'C1')
ax2.grid(True)
ax2.axis('tight')
plt.show()

# unwrapped desenvuelve la respuesta de fase, evita la discontinuidad 

# El ancho de banda digital deberia ser minimamente mas grande que la frecuencia de stop 
# La banda de rechazo no tiene sentido que empiece en Nyquist , tiene que empezar en un numero mayor, el ancho de banda tiene que ser bastante mayor que la w de stop
# En digital, el mismo filtro es de orden menor, esto se debe a la compresion de la banda de stop en pi radianes. 
# Para forzarle el orden 4 en este caso, puedo aumentar la fs
# Como se comprime el espectro, el ancho de banda es mas grande, y la relacion paso a ser lineal 

# Que sea xripple en la banda de paso y en la banda de stop, nunca excede el valor de dB, siempre oscila en esa banda (en este caso 3dB y 10dB)

#%% Ahora vamos a exigirle un poco mas a la plantilla 
# FILTRO DIGITAL (pasa bajos)
# Vuelvo a mis valores no normalizados y le paso a la funcion la frecuencia fs --> le paso la plantilla desnormalizada  
fs = 500 # Hz
wp = 70 # Hz --> es wc
ws = 100 # Hz --> es ws
gpass = 1 # dB --> es alfamax
gstop = 50 # dB --> es alfamin

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'butter', output = 'ba', fs = fs)
# Es de maxima planicidad en ambas bandas 

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cauer', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cauer)
# Es equiripple en la banda de paso y en la banda de stop 

b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby1', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cheby 1) 
# Es equiripple en la banda de paso y max planicidad (parecido a max planicidad) en banda de stop 

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby2', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cheby 2) 
# Es equiripple en la banda de stop y max planicidad (parecido a max planicidad) en banda de paso

taps = b_coeff.shape[0]

omega, resp_freq = sig.freqz(b_coeff, a = a_coeff, worN = 1024, fs = fs)

fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, tight_layout = True)
ax1, ax2 = axs
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter") # taps es la cantidad de coeficientes --> me va a dar el orden del filtro 
ax1.plot(omega, 20 * np.log10(abs(resp_freq)), 'C0')
# ax1.plot(omega, abs(resp_freq), 'C0')
ax1.set_ylabel("Amplitude in dB", color = 'C0')
ax1.grid(True)
ax1.axis('tight')
ax1.set(xlabel="Frequency in rad/sample", xlim = (0, np.pi))

# ax2 = ax1.twinx()

phase = np.unwrap(np.angle(resp_freq)) # En este caso si me conviene usar unwrap, porque evito la discontinuidad en la fase
# phase = np.angle(resp_freq)

ax2.plot(omega, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color = 'C1')
ax2.grid(True)
ax2.axis('tight')
plt.show()

# Lo que aparece en el butter abajo es porque la se llega al piso de la funcion, es decir que ahi falla
# Ajustamos la plantilla y se comporta bien, pero con muchos mas taps

#%% Hacemos un pasa banda 
# FILTRO DIGITAL (pasa banda)
# Vuelvo a mis valores no normalizados y le paso a la funcion la frecuencia fs --> le paso la plantilla desnormalizada  
fs = 500 # Hz

# Queremos filtrar la banda de paso de un ECG
wp1 = 1 # Hz
wp2 = 35 # Hz --> 30 es aprox el ancho de banda que nos dio en la TS5
ws1 = .1 # Hz \
ws2 = 45 # Hz
gpass = .5 # dB
gstop = 50 # dB 

wp = [wp1, wp2] # Comienzo y fin de la banda de paso
ws = [ws1, ws2] # Comienzo y fin de la banda de stop

b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'butter', output = 'ba', fs = fs)
# Es de maxima planicidad en ambas bandas 

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cauer', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cauer)
# Es equiripple en la banda de paso y en la banda de stop 

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby1', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cheby 1) 
# Es equiripple en la banda de paso y max planicidad (parecido a max planicidad) en banda de stop 

# b_coeff,a_coeff = sig.iirdesign(wp, ws, gpass, gstop, analog = False, ftype = 'cheby2', output = 'ba', fs = fs) # Ahora voy al mundo digital (Transferencia tipo Cheby 2) 
# Es equiripple en la banda de stop y max planicidad (parecido a max planicidad) en banda de paso

taps = b_coeff.shape[0]

omega, resp_freq = sig.freqz(b_coeff, a = a_coeff, worN = 4096, fs = fs)

fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, tight_layout = True)
ax1, ax2 = axs
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter") # taps es la cantidad de coeficientes --> me va a dar el orden del filtro 
ax1.plot(omega, 20 * np.log10(abs(resp_freq)), 'C0')
# ax1.plot(omega, abs(resp_freq), 'C0')
ax1.set_ylabel("Amplitude in dB", color = 'C0')
ax1.grid(True)
ax1.axis('tight')
ax1.set(xlabel="Frequency in rad/sample", xlim = (0, np.pi))

# ax2 = ax1.twinx()

phase = np.unwrap(np.angle(resp_freq)) # En este caso si me conviene usar unwrap, porque evito la discontinuidad en la fase
# phase = np.angle(resp_freq)

ax2.plot(omega, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color = 'C1')
ax2.grid(True)
ax2.axis('tight')
plt.show()

# Implementando este filtro con 'ba', me da muchos coeficientes (51) --> lo reemplazo por un SOS, y los 50 taps se reparten en 25 etapas
#%% Ahora, implementando el SOS 
# Para el SOS no me sirve el freqz, uso freqz_sos 
# FILTRO DIGITAL (pasa banda)
# Vuelvo a mis valores no normalizados y le paso a la funcion la frecuencia fs --> le paso la plantilla desnormalizada  
fs = 500 # Hz

wp1 = 1 # Hz
wp2 = 35 # Hz --> 30 es aprox el ancho de banda que nos dio en la TS5
ws1 = .1 # Hz \
ws2 = 45 # Hz
gpass = .5 # dB
gstop = 50 # dB 

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

fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, tight_layout = True)
ax1, ax2 = axs
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter") # taps es la cantidad de coeficientes --> me va a dar el orden del filtro 
ax1.plot(omega, 20 * np.log10(abs(resp_freq)), 'C0')
# ax1.plot(omega, abs(resp_freq), 'C0')
ax1.set_ylabel("Amplitude in dB", color = 'C0')
ax1.grid(True)
ax1.axis('tight')
ax1.set(xlabel="Frequency in rad/sample", xlim = (0, np.pi))

# ax2 = ax1.twinx()

phase = np.unwrap(np.angle(resp_freq)) # En este caso si me conviene usar unwrap, porque evito la discontinuidad en la fase
# phase = np.angle(resp_freq)

ax2.plot(omega, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color = 'C1')
ax2.grid(True)
ax2.axis('tight')
plt.show()

# Esto hace que la plantilla funcione porque bajamos las recursiones 