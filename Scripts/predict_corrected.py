import numpy as np
import matplotlib.pyplot as plt
from numpy import pi as pi
from numpy import sin as sin
from numpy import cos as cos
from scipy import signal
import statsmodels.api as sm
import pandas as pd
## UPDATE 2024 ##################################################
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA
#################################################################
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


#plt.plot(y);plt.show()
dta=pd.DataFrame(y)
## UPDATE 2024 ##################################################
arma_mod = ARIMA(y, order=(2, 0, 0), trend="n")
arma_res = arma_mod.fit()
plot_predict(arma_res, end=len(y)+100)
plt.plot(y)
plt.show()

## DEPRECATED #######################################################
# res = sm.tsa.ARMA(dta, (2, 0)).fit()
# res.plot_predict(500, 520, dynamic=False, plot_insample=False)
# plt.plot(dta)

####################################################################
# MA(1)
N =1000
b1 =0.25
e_t0 = np.random.normal(0,0.1)
e_t1 = np.random.normal(0,0.1)
y = np.append(e_t0,b1*e_t0+e_t1)  # y1 = b1*e_t0+e_t1

for i in range(N):
   e_t0 = e_t1
   e_t1 = np.random.normal(0,1)
   y = np.append(y,b1*e_t0+e_t1)


y.mean()
y.std()
plt.plot(y);plt.show()

# Predict
dta=pd.DataFrame(y)
## UPDATE 2024 ##################################################
arma_mod = ARIMA(y, order=(0, 0, 3), trend="n")
arma_res = arma_mod.fit()
plot_predict(arma_res, end=len(y)+20)
plt.plot(y)
plt.show()
print(arma_res.summary())

## DEPRECATED #######################################################
# res = sm.tsa.ARMA(dta, (0, 1)).fit()
# plt.plot(dta)
# res.plot_predict(500, 1010, dynamic=False, plot_insample=False, alpha=0.1)
# plt.show()
# plt.plot(dta)
# arma_res.plot_predict(500, 1010, dynamic=False, plot_insample=False)

# print(res.summary())

####################################################################
#ARMA(2,1)
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi as pi
from numpy import sin as sin
from numpy import cos as cos
from scipy import signal
import statsmodels.api as sm
import pandas as pd

N=10000
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
#plt.plot(y);
#plt.show()

dta=pd.DataFrame(y)
## UPDATE 2024 ##################################################
arma_mod = ARIMA(y, order=(2, 0, 1), trend="n")
arma_res = arma_mod.fit()
plot_predict(arma_res, end=len(y)+20)
plt.plot(y)
plt.show()
print(arma_res.summary())

## DEPRECATED #######################################################


# res = sm.tsa.ARMA(dta, (2, 1)).fit()
# fig, ax = plt.subplots()
# ax = dta.plot(ax=ax)
# fig=res.plot_predict(int(N-N/2), int(N+N/100), dynamic=False, ax=ax, plot_insample=False, alpha=0.1)
# plt.show()



####################################################################
#ARIMA (2,0,1)
# test first with N=1000, then N= 10000
N=1000
a1=0.35
a2=0.25
b1 =-0.35

x=np.arange(2)
e_t=np.random.normal(0,1)
y=np.append(x,a1*x[1]+a2*x[0]+e_t)

for i in range(N):
   e_0 = e_t
   e_t=np.random.normal(0,1)
   y=np.append(y,a1*y[-1]+a2*y[-2]+ b1*e_0 + e_t)

t = np.arange(int(N*0.01), step=0.01)
y=y[2:N+2]

y = y+t
y.mean()
y.std()
plt.plot(y);plt.show()



dta=pd.DataFrame(y)
## UPDATE 2024 ##################################################
arma_mod = ARIMA(y, order=(2, 0, 1), trend="n")
arma_res = arma_mod.fit()
plot_predict(arma_res, end=len(y)+5)
plt.plot(y)
plt.show()
print(arma_res.summary())

## DEPRECATED #######################################################
# Predict
# dta=pd.DataFrame(y)
# res = sm.tsa.ARMA(dta, (0, 1)).fit()
# plt.plot(dta)
# res.plot_predict(500, 1010, dynamic=False, plot_insample=False)
# plt.show()


# dta=pd.DataFrame(y)
# res = sm.tsa.ARIMA(dta, (2,0, 1)).fit()
# fig, ax = plt.subplots()
# ax = dta.plot(ax=ax)
# fig=res.plot_predict(int(N-N/2), int(N+10), dynamic=True, ax=ax, plot_insample=False)
# plt.show()


# print(res.summary())


####################################################################
#Sunspots (ARMA(3,0))

dta = sm.datasets.sunspots.load_pandas().data[['SUNACTIVITY']]
dta.index = pd.date_range(start='1700', end='2009', freq='A')

arma_mod = ARIMA(dta, order=(2, 1, 3), trend="n")
arma_res = arma_mod.fit()
plot_predict(arma_res, end='2010')
plt.show()
print(arma_res.summary())


arma_mod = ARIMA(dta, order=(3, 1, 2), trend="n")
arma_res = arma_mod.fit()
plot_predict(arma_res, end=len(dta)+20)
plt.plot(dta)
plt.show()
print(arma_res.summary())


# res = sm.tsa.ARMA(dta, (3, 0)).fit()
# fig, ax = plt.subplots()
# ax = dta.loc['1950':].plot(ax=ax)
# fig = res.plot_predict('1990', '2012', dynamic=True, ax=ax,plot_insample=False)
# plt.show()

# SARIMA ((p,d,q)(P,D,Q))
mod = sm.tsa.statespace.SARIMAX(dta, order=(2,1,0), 
                                seasonal_order=(1,1,0,12), 
                                simple_differencing=True)
res = mod.fit(disp=False)
print(res.summary())
predict_dy = res.get_prediction()
plot_predict(res, end=len(dta)+20)
plt.show()