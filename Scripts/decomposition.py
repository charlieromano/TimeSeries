import sys, getopt
import pandas as pd
import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa as tsa
import statsmodels.api as sm


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

period=21
sm.tsa.seasonal_decompose(df, model='additive', freq=period).plot()
plt.show()

