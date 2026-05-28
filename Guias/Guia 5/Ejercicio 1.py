# Ejercicio 1 - Guia 5
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal . windows as spsw

fs = 1000
N = 100
n = np . arange ( N )
x = np . cos (2* np . pi * 205 * n / fs )

freqs = np . fft . rfftfreq (N , d =1/ fs )

# Sin ventana ( rectangular ) vs Hann
for nombre , w in [(" Rectangular ", np . ones ( N ) ) ,
                   (" Hann ", spsw . hann (N , sym = False ) ) ]:
    X = np . fft . rfft ( x * w ) / N
    plt.figure ( figsize =(8 , 3) )
    plt.plot ( freqs , 20* np.log10 ( np .abs( X ) + 1e-12) , label = nombre )
    plt.axvline (205 , color ="red", linestyle =" --", label ="205 Hz real ")
    plt.xlabel (" Frecuencia [Hz]")
    plt.ylabel ("|X(f)| [dB]")
    plt.title ( f" Espectro con ventana { nombre }") 
    plt.ylim ( -80 , 0)
    plt.grid ()
    plt.legend ()
    plt.tight_layout ()
plt.show ()


