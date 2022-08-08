import csv
import numpy as np
import pandas as pd
from functions import *
import datetime as DT

"""
This script is designed to create a CSV of input data for a RNN that predict changes in price of BTC. 
We'll read in BTC price data and assign each price as a label with the preceeding price information as features.
The amount of preceeding data can be user defined. 

Known Issues: 

"""

# Set period and interval
period = "2y"
interval = "1h"
history = 120 # number of hours in the past to use as input for the RNN.
crypto = "BTC-USD"
csv_filepath = "/home/icebear/Tradebot/btc_rnn.csv"

# Get the BTC data
hist = yahoo_hist(crypto,period=period,interval=interval,prepost=True)
close = hist["Close"].values

# Open the csv file
with open(csv_filepath, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Loop through each value
    for i in range(len(close) - 1, history, -1):
        start = i - history
        stop = i
        instance = [close[i]]
        [instance.append(C) for C in close[start:stop]]
        csvwriter.writerow(instance)
# Close the CSV file
csvfile.close()
print("DONE.")