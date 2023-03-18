import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import datetime
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

np.random.seed(0)
arparams = np.array([0.75, -0.25])
maparams = np.array([0.65, 0.35])
arparams = np.r_[1, -arparams]
maparams = np.r_[1, maparams]
nobs = 1000
y = arma_generate_sample(arparams, maparams, nobs)

dates = pd.date_range("1950-1-1", freq="M", periods=nobs)
y = pd.Series(y, index=dates)
arma_mod = ARIMA(y, order=(2, 0, 2), trend="n")
arma_res = arma_mod.fit()
print(arma_res.summary())
fig, ax = plt.subplots(figsize=(10, 8))
fig = plot_predict(arma_res, start="1999-06-30", end="2001-05-31", ax=ax)
legend = ax.legend(loc="upper left")

plt.show()