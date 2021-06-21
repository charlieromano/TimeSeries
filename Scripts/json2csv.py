import pandas as pd
import json
import matplotlib.pyplot as plt

datasets = "../Datasets/"
df = pd.read_json(datasets+"TECO2.2015.2021.json")

df["ultimoPrecio"]

dt=df[['fechaHora','ultimoPrecio',]]
dt=dt.set_index(pd.DatetimeIndex(dt['fechaHora']))

dt=dt.sort_index(ascending=True)
dt['ultimoPrecio'].plot()
plt.show()



