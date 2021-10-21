import sys, getopt
import pandas as pd
import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa as tsa
import statsmodels.api as sm

period=90

inputfile="../Datasets/YPFD.2000.2021.csv"
print('Input file is "', inputfile)
df1 = pd.read_csv(inputfile)
df1['fechaHora']=pd.DatetimeIndex(df1['fechaHora'])
df1=df1[['fechaHora','ultimoPrecio']]

# Linear Regression

df=df1
nsample = len(df)
x = pd.to_numeric(pd(df1['fechaHora']))
X=x #X = np.column_stack(x)
X = sm.add_constant(X)
y = np.array(df1['ultimoPrecio'])
model = sm.OLS(y, X)
res = model.fit()
print(res.summary())

from statsmodels.sandbox.regression.predstd import wls_prediction_std
prstd, iv_l, iv_u = wls_prediction_std(res)
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, y, 'o', label="data")
ax.plot(x, res.fittedvalues, 'r--.', label="OLS")
ax.plot(x, iv_u, 'r--')
ax.plot(x, iv_l, 'r--')
ax.legend(loc='best');
plt.show()


############################################################

init="2006-01-01"
end="2018-12-31"

ts1=df1[(df1.index>=init) & (df1.index<end)]
ts2=df2[(df2.index>init) & (df2.index<end)]

len(ts1);len(ts2)
ts1, ts2 = ts1.align(ts2)
len(ts1);len(ts2)
############################################################


nsample = 50
sig = 0.5
x = np.linspace(0, 20, nsample)
X = np.column_stack((x, np.sin(x), (x-5)**2, np.ones(nsample)))
beta = [0.5, 0.5, -0.02, 5.]

y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)
res = sm.OLS(y, X).fit()

prstd, iv_l, iv_u = wls_prediction_std(res)

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(x, y, 'o', label="data")
ax.plot(x, y_true, 'b-', label="True")
ax.plot(x, res.fittedvalues, 'r--.', label="OLS")
ax.plot(x, iv_u, 'r--')
ax.plot(x, iv_l, 'r--')
ax.legend(loc='best');


