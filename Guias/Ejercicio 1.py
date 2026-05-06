import numpy as np
import matplotlib.pyplot as plt 

N = 8 
n = np.arange(N) 
x = 4 + 3 * np.sin(np.pi * n / 2) # Senal: x[n] = 4 + 3*sin(pi*n/2) 

X = np.fft.fft(x, n=N) # DFT de N puntos

k = np.arange(N) 

print("x[n]:", np.round(x, 4)) 
print("\nX[k] (complejo):") 

for kk in range(N): 
    print(f" k={kk:2d}: {X[kk]: .3f}")

fig, axes = plt.subplots(3, 1, figsize=(8, 7)) 
axes[0].stem(n, x) 
axes[0].set_title("Senal en tiempo: x[n] = 4 + 3*sin(pi*n/2)") 
axes[0].set_xlabel("n"); axes[0].set_ylabel("x[n]"); axes[0].grid(True) 
axes[1].stem(k, np.abs(X))
axes[1].set_title("DFT (N=8): Magnitud |X[k]|") 
axes[1].set_xlabel("k"); axes[1].set_ylabel("|X[k]|"); axes[1].grid(True) 
axes[2].stem(k, np.angle(X)) 
axes[2].set_title("DFT (N=8): Fase angulo(X[k]) [rad]") 
axes[2].set_xlabel("k"); axes[2].set_ylabel("Fase [rad]"); axes[2].grid(True ) 
plt.tight_layout() 
plt.show() 

# Verificacion: X[0]=32, X[2]=-12j, X[6]=+12j, resto ~0