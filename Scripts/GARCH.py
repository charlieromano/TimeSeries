import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
import arch


data = pd.read_csv('../Datasets/RIGO.2012.2021.csv', index_col=0, usecols=[1, 2], parse_dates=True)
data = data.loc[data.index.year <= 2018]
data = data.loc[data.ultimoPrecio>0]

plt.figure()
plt.plot(data)
precio = data.ultimoPrecio.values[::-1]
r_t = (np.log(precio[1:]) - np.log(precio[:-1]))*100
plt.figure()
plt.plot(r_t)
plot_acf(r_t, zero=False, title='ACF de $r_t$')
plot_pacf(r_t, zero=False, title='PACF de $r_t$', method='ywm')
plot_acf(r_t ** 2, zero=False, title='ACF de $r_t^2$')
plot_pacf(r_t ** 2, zero=False, title='PACF de $r_t^2$', method='ywm')
plt.show()
# Mirando las PACF de r^2 podemos sugerir que r^2 sigue un modelo ARMA(2,2), con lo cual la varianza
# condicional va a seguir un modelo GARCH(2,2)
garch = arch.arch_model(r_t[:-10], vol='garch', p=3, q=3, o=0, power=2)
garch_fitted = garch.fit()
garch_fitted.plot()
plt.show()
print(garch_fitted.params)
print(garch_fitted.pvalues)
print('bic: ', garch_fitted.bic)
forecast = garch_fitted.forecast(horizon=10, reindex=False)
resid = garch_fitted.resid
plt.plot(r_t[-100:]**2)
plt.plot(np.arange(90,100), forecast.variance.values.T)
plt.show()

