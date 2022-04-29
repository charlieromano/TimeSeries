import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Parameters
N = 100
mu, sigma = 0, 0.1
bins = 25
degFreedom = 9
col=0

# random values data series
X = np.random.normal(mu, sigma, N)

# Random walk
Y = np.cumsum(X)

plt.plot(X)
plt.plot(Y)
plt.legend(['X','Y'])
plt.show()

