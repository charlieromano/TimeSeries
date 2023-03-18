import numpy as np
import pandas as pd

# list  1
a = [2, 3, 2.7, 3.2, 4.1]
 
# list 2
b = [10, 14, 12, 15, 20]

# storing average of a
av_a = sum(a)/len(a)

# storing average of b
av_b = sum(b)/len(b)

# making series from list a
a = pd.Series(a)

# making series from list b
b = pd.Series(b)

# covariance through pandas method
covar = a.cov(b)

print("series a:", a)
print("series b:", b)
print("Results from Pandas method: ", covar)

