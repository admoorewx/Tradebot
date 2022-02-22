from functions import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from alpaca import all_avail_stocks

assets = all_avail_stocks()
dict = {}
for i, asset in enumerate(assets):
    try:
        dict[asset.symbol] = yahoo_price(asset.symbol)
    except:
        print(f'Had trouble finding data for {asset.symbol}')

df = pd.Dataframe(dict)

plt.figure()
sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.savefig("correlation_heatmap.png")