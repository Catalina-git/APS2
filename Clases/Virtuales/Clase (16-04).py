import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

this_order = 2

z,p,k = sig.buttap(this_order)

# eps = np.sqrt( 10**(this_ripple/10) - 1 ) --> epsilon --> como es butter, el epsilon vale 1
eps = 1

num, den = sig.zpk2tf(z,p,k) # son los coeficientes del numerado (num) y del denominador (den)

# z,p,k = sig.tf2zpk(num, den)






