import numpy as np
import matplotlib.pyplot as plt
from numpy import pi as pi
from numpy import sin as sin
from scipy import signal

# example 
N=1000
f=50
T=1/f
t=np.arange(N)
Phi=np.random.normal(10,1,N)
X=sin(2*pi*f*t/N)
Y=sin(2*pi*f*t/N + Phi)

plt.plot(Phi)
plt.plot(X)
plt.plot(Y)
plt.legend(['$\Phi$','$X_t$','$Y_t$'])
plt.show()

#F, Pxx_den = signal.periodogram(X,N)
#plt.semilogy(F, Pxx_den)
G, Pyy_den = signal.periodogram(Y,N)
plt.plot(G, Pyy_den)
#plt.legend(['$F(X)$','$F(Y)$'])
plt.show()
