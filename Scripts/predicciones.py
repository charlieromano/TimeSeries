import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import datetime
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

n=100
N=10000
mu=np.repeat(np.sin(np.arange(n)),int(N/n))
X=np.random.normal(5,25,int(N))
Y=X+mu
plt.plot(Y)
plt.plot(mu,color='red')
plt.title("Ciclic trend")
plt.legend(['$Y_t$','$\mu_t$'])
plt.show()


ma2 = ARIMA(Y, order=(0, 0, 2), trend="n")
ma2_res = ma2.fit()
print(ar1_res.summary())

fig, ax = plt.subplots(figsize=(10, 8))
fig = plot_predict(ma2_res, start=1,end=2*N, ax=ax)
plt.plot(Y)
legend = ax.legend(loc="upper left")
plt.show()