import pandas as pd
import matplotlib.pyplot as plt
from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf
from datetime import datetime

inputfile = "../Datasets/TECO2.2000.2021.csv"


ts = pd.read_csv(inputfile, header=0, index_col=0, squeeze=True)
ts.fechaHora = pd.to_datetime(ts.fechaHora)
ts.fechaHora=pd.to_datetime(ts.fechaHora).dt.date
ts.fechaHora=pd.DatetimeIndex(ts.fechaHora)
ts=ts.sort_index(ascending=False)
#ts.fechaHora=pd.to_numeric(pd.to_datetime(ts.fechaHora))
plot_acf(ts.ultimoPrecio.values, lags=12*4*7)
plt.show()


inputfile = "../Datasets/S1MME_week43.csv"
ts = pd.read_csv(inputfile)
plot_acf(ts.S1_mode_combined_attach_request_times_SEQ.values, lags=48)
plt.show()

