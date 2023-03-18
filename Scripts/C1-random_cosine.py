import numpy as np
import matplotlib.pyplot as plt
N=40
T=12
t=np.arange(N)
Phi=np.random.rand(5)
for phi in Phi:
    plt.plot(np.cos(2*np.pi*(t/T + phi)), label='Phi='+str(np.round(phi,2)))

plt.ylabel('y')
plt.xlabel('t')
plt.legend()
plt.show()
