###############################################################
## Periodogram
###############################################################

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

Phi2=np.random.normal(0,1,N)
f2=33
Y2=sin(2*pi*f2*t/N + Phi2)

Yt=Y+Y2


plt.plot(Phi)
plt.plot(X)
plt.plot(Y)
plt.legend(['$\Phi$','$X_t$','$Y_t$'])
plt.show()

#F, Pxx_den = signal.periodogram(X,N)
#plt.semilogy(F, Pxx_den)
G, Pyy_den = signal.periodogram(Phi,N)
plt.plot(G, Pyy_den)
#plt.legend(['$F(X)$','$F(Y)$'])
plt.show()




####################################################################



# AR(1)
N=1000
a1=0.5
sigma=0.1
x=np.arange(1)
y=np.append(x,a1*x+np.random.normal(0,sigma))

for i in range(N):
   y=np.append(y,a1*y[-1]+np.random.normal(0,sigma))



G, Pyy_den = signal.periodogram(y,N)
plt.plot(G, Pyy_den)
plt.show()

plt.plot(y);plt.show()


####################################################################
# AR(2)
N=1000
a1=0.4
a2=0.35
x=np.arange(2)
e_t=np.random.normal(0,1)
y=np.append(x,a1*x[1]+a2*x[0]+e_t)

for i in range(N):
   e_t=np.random.normal(0,1)
   y=np.append(y,a1*y[-1]+a2*y[-2]+e_t)


plt.plot(y);plt.show()

G, Pyy_den = signal.periodogram(y,N)
plt.plot(G, Pyy_den)
plt.show()


####################################################################



dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]
sm.graphics.tsa.plot_pacf(dta.values.squeeze(), lags=40, method="ywm")
#plt.show()

p=2
d=1
q=1
P=0
D=0
Q=1
S=12

model = ARIMA(dta.values, order=(p,d,q), seasonal_order=(P,D,Q,S))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
#plt.show()
# density plot of residuals
residuals.plot(kind='kde')
#plt.show()
# summary stats of residuals
print(residuals.describe())

sm.qqplot((residuals-residuals.mean())/residuals.std(),  line ='45')
plt.title('SARIMA(5,1,0)(1,1,1)12')
plt.show()


####################################################################


