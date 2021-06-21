import pandas as pd
import json
import matplotlib.pyplot as plt

datasets = "../Datasets/"
df = pd.read_json(datasets+"TECO2.2015.2021.json")

df["ultimoPrecio"]
df["fechaHora"]=pd.to_datetime(df["fechaHora"])

df.set_index('fechaHora')

df["ultimoPrecio"].plot()
