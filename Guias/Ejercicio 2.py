import numpy as np 
import matplotlib.pyplot as plt 

L = 3 # longitud del pulso 
N_zp = 100000 # zero-padding grande para aproximar la DTFT continua 

# Senal: x[n] = delta[n] + delta[n-1] + delta[n-2] 
x = np.zeros(N_zp) 
x[:L] = 1.0 
X = np.fft.fft(x, n=N_zp) 
omega_k = 2 * np.pi * np.arange(N_zp) / N_zp 

# DTFT teorica para L=3: |1 + e^{-jw} + e^{-j2w}| 
omega_c = np.linspace(0, 2 * np.pi, 2000, endpoint=False) 
dtft_mag = np.abs(1 + np.exp(-1j * omega_c) + np.exp(-2j * omega_c)) 

# Graficos
plt.figure(figsize=(8, 4))
plt.plot(omega_c, dtft_mag, label="DTFT teorica (L=3)") 
plt.plot(omega_k, np.abs(X), 'o', ms=2, label=f"Muestras DFT (N={N_zp})")
plt.title("DTFT vs DFT con zero-padding") 
plt.xlabel("omega [rad]"); plt.ylabel("Magnitud") 
plt.legend(); plt.grid(True) 
plt.show()
