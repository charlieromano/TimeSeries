import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from scipy.linalg import hankel

def ARMAtoMA(ar, ma, max_lag):
    p = len(ar)
    q = len(ma)
    ma = np.concatenate((ma, np.zeros(max_lag-q)))
    psi = np.ones(max_lag)
    for i in range(max_lag):
        tmp = -ma[i]
        for j in range(min(i+1,p)):
            if i-j-1 >= 0:
                tmp += ar[j] * psi[i-j-1]
            else:
                tmp += ar[j]
        psi[i] = tmp;
    return np.concatenate(([1],psi))

def ARMAtoAR(ar, ma, max_lag):
    return ARMAtoMA(-ma, -ar, max_lag)


# definimos una función que calcule el estimador para el test
def hip_testing_IO(model, alpha):
    """model: modelo ARMA o SARIMAX previamente entrenado
       alpha: nivel de significación del test
    """
    try:
        q = len(model.maparams)
    except:
        q = 0
    sigma = np.sqrt(np.pi / 2) * np.nanmean(np.abs(model.resid[q:]))
    estimador = model.resid[q:] / sigma  # np.sqrt(model.cov_params()['sigma2'].sigma2)
    umbral = stats.norm.ppf(1 - alpha / 2 / (model.nobs - q))
    return np.abs(estimador), np.where(np.abs(estimador) > umbral)[0]


np.random.seed(3)
arparams = np.array([0.8])
maparams = np.array([-0.7])
data = arma_generate_sample(ar=np.r_[1, -arparams], ma=np.r_[1, maparams], nsample=200, burnin=158)
data[10] = 5
plt.plot(data)

sarima  = ARIMA(endog=data, order=(1,0,1), trend='n', enforce_invertibility=True, enforce_stationarity=True)
sarima_fit = sarima.fit()
l1 = hip_testing_IO(sarima_fit, 0.05)
sarima_fit.summary()

print('l1: ',l1[1])













