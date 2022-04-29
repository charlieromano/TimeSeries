import pandas as pd
import matplotlib.pyplot as plt
from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]
sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40)
plt.show()

dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]
sm.graphics.tsa.plot_pacf(dta.values.squeeze(), lags=40, method="ywm")
plt.show()

model = ARIMA(dta.values, order=(5,1,0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())

sm.qqplot((residuals-residuals.mean())/residuals.std(),  line ='45')
plt.title('ARIMA(5,1,0)')
plt.show()


#########################################################################

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
