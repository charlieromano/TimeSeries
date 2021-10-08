import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Parameters
N = 100000
mu, sigma = 0, 0.1
bins = 25
degFreedom = 9

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

# adding datetime index
col=0
time = pd.date_range('2021-10-08', periods=N, freq='s')
df.insert(col,"Datetime", time, True)
df.set_index('Datetime')
df.set_index('Datetime').plot()
plt.show()

# dataframe to csv
df.to_csv('../Datasets/randomValues.csv')



