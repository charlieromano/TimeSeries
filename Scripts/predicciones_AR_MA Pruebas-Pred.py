import numpy as np
import matplotlib.pyplot as plt
from pandas import datetime
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
####################################################################
np.random.seed(1)
# AR(2)
N=150
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
ar2 = SARIMAX(y, order=(2, 0, 0))
ar2_res = ar2.fit()
print(ar2_res.summary())

fig, ax = plt.subplots(figsize=(10, 8))
fig = plt.plot(ar2_res.forecast(l)) #plot_predict(ar2_res, start=0,end=N+l, dynamic=N+2, ax=ax)
plt.plot(y)
legend = ax.legend(loc="upper left")
plt.show()

####################################################################



# AR(1)
N=150
a1=0.5
sigma=0.1
x=np.arange(1)
y=np.append(x,a1*x+np.random.normal(0,sigma))

for i in range(N):
   y=np.append(y,a1*y[-1]+np.random.normal(0,sigma))


y.mean()
y.std()
plt.plot(y);plt.show()

l=10
ar1 = SARIMAX(y, order=(1, 0, 0))
ar1_res = ar1.fit()
print(ar1_res.summary())

fig, ax = plt.subplots(figsize=(10, 8))
fig = plt.plot(ar1_res.forecast(l)) #plot_predict(ar1_res, start=0,end=N+l, dynamic=N+2, ax=ax)
plt.plot(y)
legend = ax.legend(loc="upper left")
plt.show()


####################################################################
# MA(1)
N =150
b1 =0.25
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
ma1 = SARIMAX(y, order=(0, 0, 1))
ma1_res = ma1.fit()
print(ma1_res.summary())

fig, ax = plt.subplots(figsize=(10, 8))
fig = plt.plot(ma1_res.forecast(l)) #plt.plot(SARIMAX.forecast(l)) #plot_predict(ma1_res, start=0,end=N+l, dynamic=N+2, ax=ax)
plt.plot(y)
legend = ax.legend(loc="upper left")
plt.show()

y_ma1=y
####################################################################
#ARMA

N=150
a1=0.4
a2=0.3
b1 =-0.3

x=np.arange(2)
e_t=np.random.normal(0,1)
y=np.append(x,a1*x[1]+a2*x[0]+e_t)

for i in range(N):
   e_0 = e_t
   e_t=np.random.normal(0,1)
   y=np.append(y,a1*y[-1]+a2*y[-2]+ b1*e_0 + e_t)


y.mean()
y.std()
#plt.plot(y);plt.show()


# Predict
y_arma = SARIMAX(y, order=(2, 0, 1))
y_arma_res = y_arma.fit()
print(y_arma_res.summary())

fig, ax = plt.subplots(figsize=(10, 8))
fig = plt.plot(y_arma_res.forecast(l)) #plot_predict(y_arma_res, start=0,end=N+l, dynamic=N+2, ax=ax )
plt.plot(y)
legend = ax.legend(loc="upper left")
plt.show()



####################################################################
#ARIMA

N=150
a1=0.35
a2=0.25
b1 =-0.35
l=20
x=np.arange(2)
e_t=np.random.normal(0,1)
y=np.append(x,a1*x[1]+a2*x[0]+e_t)

for i in range(N):
   e_0 = e_t
   e_t=np.random.normal(0,1)
   y=np.append(y,a1*y[-1]+a2*y[-2]+ b1*e_0 + e_t)

t = np.arange(N)*0.01

y=y[2:N+2]

y = y+t[2:N+2]
y.mean()
y.std()
#plt.plot(y);plt.show()


# Predict
y_arima = SARIMAX(y, order=(2, 1, 1), trend='t')
y_arima_res = y_arima.fit()
print(y_arma_res.summary())

forecasts_0 = y_arima_res.forecast(l)

forecasts = [y_arima_res.forecast()]
for t in range(l):
    # Update the results by appending the next observation
    y_arima_res = y_arima_res.append(forecasts[-1], refit=False)

    # Save the new set of forecasts
    forecasts.append(y_arima_res.forecast(steps=1))

# Combine all forecasts into a dataframe
forecasts = np.append(forecasts[0], forecasts[1:])


fig, ax = plt.subplots(figsize=(10, 8))
fig = plt.plot(np.arange(N,N+l+1), forecasts) #plot_predict(y_arma_res, start=0,end=N+l,  dynamic=N, ax=ax)
plt.plot(y)
plt.plot(np.arange(N,N+l), forecasts_0)
legend = ax.legend(loc="upper left")
plt.show()

