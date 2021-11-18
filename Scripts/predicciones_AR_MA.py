import numpy as np
import matplotlib.pyplot as plt
from pandas import datetime
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
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


y.mean()
y.std()
#plt.plot(y);plt.show()

# Predict
l=20
ar2 = ARIMA(y, order=(2, 0, 0))
ar2_res = ar2.fit()
print(ar2_res.summary())

fig, ax = plt.subplots(figsize=(10, 8))
fig = plot_predict(ar2_res, start=0,end=N+l, ax=ax)
plt.plot(y)
legend = ax.legend(loc="upper left")
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


y.mean()
y.std()
plt.plot(y);plt.show()

ar1 = ARIMA(y, order=(1, 0, 0))
ar1_res = ar1.fit()
print(ar1_res.summary())

fig, ax = plt.subplots(figsize=(10, 8))
fig = plot_predict(ar1_res, start=0,end=N+l, ax=ax)
plt.plot(y)
legend = ax.legend(loc="upper left")
plt.show()


####################################################################
# MA(1)
N =10000
b1 =0.5
e_t0 = np.random.normal(0,1)
e_t1 = np.random.normal(0,1)
y = np.append(e_t0,b1*e_t0+e_t1)  # y1 = b1*e_t0+e_t1

for i in range(N):
   e_t0 = e_t1
   e_t1 = np.random.normal(0,1)
   y = np.append(y,b1*e_t0+e_t1)


y.mean()
y.std()
plt.plot(y);plt.show()

# Predict
ma1 = ARIMA(y, order=(0, 0, 1))
ma1_res = ma1.fit()
print(ma1_res.summary())

fig, ax = plt.subplots(figsize=(10, 8))
fig = plot_predict(ma1_res, start=1,end=1010, ax=ax)
plt.plot(y)
legend = ax.legend(loc="upper left")
plt.show()
####################################################################

