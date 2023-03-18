import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

inputfile = "../Datasets/S1MME_week44.csv"
df = pd.read_csv(inputfile)
df = df.loc[df.NENAME=='MME1BEL']
sig=df.S1_mode_combined_attach_request_times_SEQ

N=24 #hoursCLI
sample=4 #sampled period number

dataframe = pd.Series(sig)
ts=pd.DataFrame(dataframe.values)
rows=int(len(ts)/N)
data = ts.values.reshape(rows,N)

#####################################################################
## estimación por valor medio de cada hora
#####################################################################

betas=data.mean(axis=0)
plt.plot(betas)
plt.plot(data[sample,:])
plt.plot(data[sample,:]-betas)
plt.legend(['betas','data','error'])
plt.show()

#####################################################################
## repetir para todo el largo de la serie
#####################################################################

est=np.tile(betas,rows)

plt.plot(sig)
plt.plot(est)
plt.plot(sig-est)
plt.legend(['sig','est','sig-est'])
plt.show()

#####################################################################
## ejemplo diferenciando
#####################################################################

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

#repito la estimacion pero para la diferenciada
N=24 #hours
dataframe = pd.Series(pd.concat([pd.Series(diff[0]),pd.Series(diff)], axis=0))
ts=pd.DataFrame(dataframe.values)
rows=int(len(ts)/N)
data = ts.values.reshape(rows,N)

betas=data.mean(axis=0)
plt.plot(betas)
plt.plot(data[sample,:])
plt.plot(data[sample,:]-betas)
plt.legend(['betas','data','data-betas'])
plt.show()

#####################################################################
## repetir para todo el largo de la serie
#####################################################################

est=np.tile(betas,rows)

plt.plot(sig)
plt.plot(est)
plt.plot(sig-est)
plt.legend(['sig','est','sig-est'])
plt.show()
