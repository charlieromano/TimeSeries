import sys, getopt
import pandas as pd
import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa as tsa
import statsmodels.api as sm

period=90

inputfile="../Datasets/RIGO.2012.2021.csv"
print('Input file is "', inputfile)
df = pd.read_csv(inputfile)
df = df.set_index(pd.DatetimeIndex(df['fechaHora']))
df.sort_index(ascending=True, inplace=True)
df=df['ultimoPrecio']

df.plot()
plt.title("timeseries: "+inputfile)
plt.xlabel("time")
plt.ylabel("AR$ ")
plt.grid()
plt.show()


# Example
sm.tsa.seasonal_decompose(df, model='additive', freq=period).plot()
plt.show()

# Additive
addit=sm.tsa.seasonal_decompose(df, model='additive', freq=period)
addit.plot()

# Multiplicative
mult = sm.tsa.seasonal_decompose(df, model='multiplicative', freq=period)
mult.plot(); plt.show()

# Trend
period=7
addit=sm.tsa.seasonal_decompose(df, model='additive', freq=period)
mult = sm.tsa.seasonal_decompose(df, model='multiplicative', freq=period)
fig, axs = plt.subplots(3)
fig.suptitle('Trends: additive vs. multiplicative')
axs[0].plot(addit.trend)
axs[1].plot(mult.trend)
axs[2].plot(addit.trend)
axs[2].plot(mult.trend)
plt.show()

period=90
addit=sm.tsa.seasonal_decompose(df, model='additive', freq=period)
mult = sm.tsa.seasonal_decompose(df, model='multiplicative', freq=period)
plt.plot(addit.trend, color="blue")
plt.plot(mult.trend, color="red")
plt.show()
