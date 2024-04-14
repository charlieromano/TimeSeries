import numpy as np
import matplotlib.pyplot as plt
from numpy import pi as pi
from numpy import sin as sin
from numpy import cos as cos
from scipy import signal

# example 1a 
N=1000
f=50
T=1/f
t=np.arange(N)
Phi=np.random.normal(0,1,N)
X=sin(2*pi*f*t/N)
Y=sin(2*pi*f*t/N + Phi)

#plt.plot(Phi)
#plt.plot(X)
plt.plot(Y)
plt.legend(['$\Phi$','$X_t$','$Y_t$'])
plt.show()

#F, Pxx_den = signal.periodogram(X,N)
#plt.semilogy(F, Pxx_den)
G, Pyy_den = signal.periodogram(Y,N)
plt.plot(G, Pyy_den)
#plt.legend(['$F(X)$','$F(Y)$'])
plt.show()


#example 1b
N=10000
f1=50
f2=1e3

t=np.arange(N)
Phi1=np.random.normal(0,5,N)
Phi2=np.random.normal(0,5,N)
X=sin(2*pi*f1*t/N)*cos(2*pi*f2*t/N)
Y=sin(2*pi*f1*t/N + Phi1)*cos(2*pi*f2*t/N+Phi2)

#plt.plot(Phi1)
#plt.plot(X)
plt.plot(Y)
plt.legend(['$\Phi$','$X_t$','$Y_t$'])
plt.show()

F, Pxx_den = signal.periodogram(X,N)
plt.semilogy(F, Pxx_den)
plt.show()
G, Pyy_den = signal.periodogram(Y,N)
plt.plot(G, Pyy_den)
#plt.legend(['$F(X)$','$F(Y)$'])
plt.show()

# example 1c
T1=24
T2=7*24
T3=90*24
N=2*T3
A=0.9
B=0.3
C=0.66

t=np.arange(N)
Phi1=np.random.normal(0,1,N)
Phi2=np.random.normal(0,1,N)
X=A*sin(2*pi*t/T1)+B*sin(2*pi*t/T2)+C*sin(2*pi*t/T3)
#plt.plot(Phi1)
plt.plot(t,X*X)
#plt.legend(['$\Phi$','$X_t$','$Y_t$'])
plt.show()

F, Pxx_den = signal.periodogram(X,N)
plt.plot(F, Pxx_den)
plt.show()
#G, Pyy_den = signal.periodogram(Y,N)
#plt.plot(G, Pyy_den)
#plt.legend(['$F(X)$','$F(Y)$'])
#plt.grid()
#plt.show()
import pandas as pd
inputfile = "Datasets/S1MME_week43.csv"
df = pd.read_csv(inputfile)
Y=df.S1_mode_combined_attach_success_times_SEQ
N=int(len(Y)/2)

G, Pyy_den = signal.periodogram(Y,N)
plt.subplot(2,1,1)
plt.plot(Y)
plt.subplot(2,1,2)
plt.plot(G, Pyy_den)
plt.show()

dataset=Y
interval=1
diff = list()
for i in range(interval, len(dataset)):
	value = dataset[i] - dataset[i - interval]
	diff.append(value)

plt.plot(Y)
plt.plot(diff)
plt.legend(['$Y_t$','$\\nabla Y_t$'])
plt.show()

N=int(len(diff))

G, Pyy_den = signal.periodogram(Y,N)
plt.subplot(2,1,1)
plt.plot(diff)
plt.subplot(2,1,2)
plt.plot(G, Pyy_den)
plt.grid()
plt.show()

plt.subplot(3,1,1) 
plot_acf(Y)#, zero=False, title='ACF de $r_t$', ax=ax[0,0])
plt.subplot(3,1,2) 
plot_pacf(Y)#, zero=False, title='ACF de $r_t$', ax=ax[0,0])
plt.subplot(3,1,3) 
plt.plot(G, Pyy_den)
plt.show()

# iterar, agregar más componentes
f1=21
f2=42
A=12.5*1e4
B=3.85*1e4

t=np.arange(N)
Y_=A*sin(2*pi*f2*t/N)+B*sin(2*pi*f1*t/N)

#plt.plot(Y)
plt.plot(Y-200e3)
plt.plot(Y_)
plt.grid()
#plt.legend(['$Y_t$','$\\nabla Y_t$','$\\hat{Y}_t$'])
plt.legend(['$Y_t-2e5$','$\\hat{Y}_t$'])
plt.show()


