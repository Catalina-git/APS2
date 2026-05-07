import numpy as np
import matplotlib.pyplot as plt

# Parámetros
N = 1000   # cantidad de muestras
R = 200    # cantidad de realizaciones
fs = N     # frecuencia de muestreo
df = fs / N
a0 = np.sqrt(2)    # amplitud

# --- Tiempo ---
t = np.arange(N) / fs   # vector de tiempo base (N,)
# Cada fila es una realización, columnas son tiempo
t_mat = np.tile(t, (R,1))   # (200, 1000)

# --- Frecuencia ---
f = np.arange(N) * df   # vector de frecuencias base (N,)
# Cada fila es una realización, columnas son frecuencia
f_mat = np.tile(f, (R,1))   # (200, 1000)

print("t_mat shape:", t_mat.shape)
print("f_mat shape:", f_mat.shape)

#%% Frecuencias aumentan para la derecha y el tiempo para abajo

# Matriz tiempo N x R (tiempo hacia abajo, realizaciones en columnas)
t_mat = np.tile(t.reshape(-1,1), (1,R))   # (1000, 200)

# --- Frecuencia ---
# Vector de frecuencias base (N puntos)
f = np.arange(N) * df   # (N,)

# Matriz frecuencia R x N (frecuencia hacia la derecha, realizaciones en filas)
f_mat = np.tile(f, (R,1))   # (200, 1000)

print("t_mat shape:", t_mat.shape)  # (1000, 200)
print("f_mat shape:", f_mat.shape)  # (200, 1000)

# --- Frecuencia aleatoria fr ~ U(-2,2) ---
fr = np.random.uniform(-2,2, size=R)   # una fr distinta por realización
Omega0 = np.pi/2
Omega1 = Omega0 + fr * (2*np.pi/N)     # frecuencia perturbada

# --- Señal senoidal ---
# Cada columna de t_mat es tiempo, cada fila de Omega1 es una realización
# Usamos broadcasting para generar la matriz senoidal
s_mat = a0 * np.sin(Omega1.reshape(1,-1) * t_mat)

print("s_mat shape:", s_mat.shape)  # (1000, 200)

# --- Graficar algunas realizaciones ---
plt.figure()
plt.plot(t_mat, s_mat)   # cada columna se grafica como una curva
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Señales senoidales con frecuencias aleatorias")
plt.grid(True)
plt.show()

X_mat = (1/N) * np.fft.fft(s_mat, axis=0)

X_mat_modulo = np.abs(X_mat)

X_mat_dB = 10 * np.log10(2 * (X_mat_modulo)**2)

plt.figure()
plt.plot(f, X_mat_dB)   
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [dB]")
plt.title("Densidad espectral de potencia de las realizaciones")
plt.xlim([0, fs/2])       # solo hasta Nyquist
plt.grid(True)
plt.show()
