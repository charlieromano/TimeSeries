import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

inputfile = "../Datasets/S1MME_week43.csv"
df = pd.read_csv(inputfile)
sig=df.S1_mode_combined_attach_request_times_SEQ
df.plot()
plt.title("timeseries: "+inputfile)
plt.xlabel("time")
plt.grid()
plt.show()


dataset=sig
interval=1
diff = list()
for i in range(interval, len(dataset)):
	value = dataset[i] - dataset[i - interval]
	diff.append(value)

plt.plot(sig)
plt.plot(diff)
plt.legend(['sig','diff'])
plt.show()
