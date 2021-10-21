import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Parameters
N = 100000
mu, sigma = 0, 0.1
bins = 25
degFreedom = 9
col=0

# random values data series
X = np.random.normal(mu, sigma, N)

plt.plot(X)
plt.show()

count, bins, ignored = plt.hist(X, bins, density=True)
plt.plot(bins, 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins-mu)**2/(2*sigma**2)), linewidth=2, color='r')
plt.show()

# random values with t-student distribution series
Y = np.random.standard_t(degFreedom, N)
plt.plot(Y)
plt.show()

plt.hist(Y,bins=bins)
plt.show()


# using Dataframe structure
df=pd.DataFrame({'X':X,'Y':Y})
df.plot()
plt.show()

# comparing timeseries
## timeseries
fig, (ax1, ax2) = plt.subplots(2)
fig.subtitle('X vs. Y timeseries')
ax1.plot(X)
ax2.plot(Y)
plt.show()
## histograms
fig, (ax1, ax2) = plt.subplots(2)
fig.subtitle('X vs. Y timeseries')
ax1.hist(X,bins=30)
ax2.hist(Y,bins=20)
plt.show()
## stats
df.X.describe
df.Y.describe()
dt = pd.DataFrame(df.X.describe())
dt=pd.concat([df.X.describe(), df.Y.describe()],axis=1)

# adding datetime index
time = pd.date_range('2021-10-08', periods=N, freq='s')
df.insert(col,"Datetime", time, True)
df.set_index('Datetime')
df.set_index('Datetime').plot()
plt.show()

# dataframe to csv
df.to_csv('../Datasets/randomValues.csv')



