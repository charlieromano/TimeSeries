import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from causalimpact import CausalImpact
from statsmodels.tsa.seasonal import seasonal_decompose

# Get data
inputfile = "../Datasets/S1MME_week43.csv"
df = pd.read_csv(inputfile)
df = df.rename(columns={'MyDay': 'date', 'S1_mode_combined_attach_success_times_SEQ': 'close'})
df=df[df.NENAME=='MME1BEL']
df=df.set_index(df.date)

# Plot close price, 21d and 252 roll avg.
df.close.plot();plt.show()
df['close'].plot(lw=2., figsize=(14,6), label='Close Price',c='royalblue')
df['close'].rolling(21).mean().plot(lw=1.5, label='Roll_21d',c='orange')
df['close'].rolling(252).mean().plot(lw=1.5, label='Roll_252d', c='salmon')
plt.title('...')
plt.ylabel('Close Price, $/MT')
plt.grid(); plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df['close'], lw=2., label='Close Price')
ax.plot(df['close'][df.index == '2019-01-25'], 
        marker='o', color='r', markersize=8,
        label='Vale Dam Collapse 25 Jan 19')
ax.plot(df['close'][df.index == '2019-05-17'], 
        marker='o', color='r', markersize=8,
        label='Vale Second Warning, 17 May 19')
plt.title('Spot Iron Ore Historical ($/MT), 2015-2020')
plt.ylabel('Close Price, $/MT')
plt.grid(); plt.legend()


t1=22
t2=32

df['close_252d_rolling'] = df['close'].rolling(t1).mean()
df['close_21d_rolling'] = df['close'].rolling(t2).mean() 

pre_period = [df.index[0], '2020/10/19 22:00'] # Define pre-event period
post_period = ['2020/10/20 01:00', df.index[-1]] # Define post-event period


pre_period_df = df[df.index <= '2020/10/19 22:00']
post_period_df = df[df.index >= '2020/10/20 01:00']
print('Pre-Event Statistics')
print(pre_period_df.describe())
print('Post-Event Statistics')
print(post_period_df.describe())


ci = CausalImpact(df['close'], pre_period, post_period)
# <<<<<<< HEAD
# ##########################
# # DESDE ACA FALTA ARREGLAR
# #########################
# ci#n.plot(figsize=(12, 6))
# =======

# ci.plot(figsize=(12, 6))
# >>>>>>> bdb03bf2f7c264611746c7c34ad6e7e85bf30137
# ci.plot(panels=['original', 'pointwise'], figsize=(12, 8))
# print(ci.summary())

# ci.trained_model.params
# print(ci.trained_model.summary())
# _ = ci.trained_model.plot_diagnostics(figsize=(14,6))

# ci.trained_model.specification

# df['close'].plot(figsize=(12,4))

# >>>>>>> 1935a5e774190cfb61c4f69560f1222e1474e001
