from alpaca_trade_api.rest import REST, TimeFrame

from scipy import interpolate
from scipy.optimize import curve_fit, leastsq
from scipy.signal import correlation_lags, correlate
from scipy.stats import spearmanr

import datetime as DT
import yfinance as YF
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns; sns.set()
import pandas as pd
import datetime

def root_mean_squared_error(truth,est):
    truth = np.asarray(truth)
    est = np.asarray(est)
    if len(truth) == len(est):
        return np.sqrt((np.sum((truth-est))**2.0)/len(truth))
    else:
        print(f'ERROR: input arrays are not the same length! ({len(truth)} and {len(est)})')
        return None

def delta_to_binary(x):
    bins = []
    for X in x:
        if X > 0.0:
            bins.append(1)
        else:
            bins.append(0)
    return bins

def discretize_delta(x):
    thres = np.std(x)
    cats = []
    for i in x:
        if i == 0.0:
            cats.append(0)
        elif i < 0:
            if i < -thres:
                cats.append(1)
            else:
                cats.append(2)
        else:
            if i < thres:
                cats.append(3)
            else:
                cats.append(4)
    return cats

def RSI(x,interval=14):
    interval = interval + 1
    RSI = []
    for i in range(len(x)):
        if i < interval:
            if i == 0:
                start = i
                end = 1
            else:
                start = 0
                end = i
        else:
            start = i - interval
            end = i

        tmp = np.asarray(x[start:end])
        pos_vals = np.where(tmp > 0.0, tmp, 0.0)
        neg_vals = np.where(tmp < 0.0, np.abs(tmp), 0.0)
        pos_mean = np.nanmean(pos_vals)
        neg_mean = np.nanmean(neg_vals)
        if neg_mean == 0.0:
            neg_mean = 1.0
        if pos_mean == 0.0:
            pos_mean = 1.0
        RS = pos_mean/neg_mean
        RSI.append(100.0 - (100.0/(1.0+RS)))

    return RSI

def momentum(x,interval=10):
    mo = []
    for i in range(len(x)):
        if i < interval:
            start = 0
            end = i
        else:
            start = i - interval
            end = i
        mo.append(x[end] - x[start])
    return mo

def running_average(data,window_size,fill=False):
    window = np.ones(int(window_size))/float(window_size)
    if fill:
        sma = np.convolve(data,window,'valid')
        while len(sma) < len(data):
            sma = np.insert(sma,0,np.nan)
        return sma

    else:
        return np.convolve(data,window,'valid')

def delta(x):
    deltas = [x[i+1] - x[i] for i in range(0,len(x)-1)]
    deltas.insert(0,0.0)
    return deltas

def zero_freq(x):
    count = 0
    if x[0] > 0:
        positive = True
    else:
        positive = False
    for i in x:
        if positive and i < 0:
            positive = False
            count = count + 1
        elif not positive and i > 0:
            positive = True
            count = count + 1
        else:
            pass
    return count

def normalize(x):
    mx = np.nanmax(x)
    mn = np.nanmin(x)
    return [(i - mn)/(mx-mn) for i in x]

def time_lagged_corr(col1, col2, lag=0):
    # tempdf = pd.Dataframe()
    # tempdf[col1] = df[col1]
    # tempdf[col1] = df[col2]
    if lag == 0:
        return spearmanr(col1,col2)[0]
    newcol = []
    for i in range(0,len(col1)):
        try:
            newcol.append(col1[i-lag])
        except:
            newcol.append(np.nan)
    # get inds of valid data
    inds = np.where(np.isnan(newcol) == False)[0]
    newcol = np.asarray(newcol)
    return spearmanr(newcol[inds],col2[inds])[0]

def yahoo_hist(stock,period="1y",interval="1d"):
    """
    Gets the open, close, high, low, volume, dividens and splits for input stock ticker.
    """
    dat = YF.Ticker(stock)
    hist = dat.history(period=period,interval=interval)
    return hist

def yahoo_price(stock,period="1y",interval="1d"):
    """
    Returns just the closing stock price of the input stock ticker.
    """
    dat = YF.Ticker(stock)
    return dat.history(period=period,interval=interval)["Close"]

def clean_data(X,Y):
    """
    Use scipy interpolate 1D to perform a cubic interpolation to fill NaN values.
    Note: this will extrapolate if NaNs are found at the start or end of "Y".
    """
    inds_to_fill = np.where(np.isnan(Y))[0]
    valid_X = [x for i,x in enumerate(X) if not np.isnan(Y[i])]
    valid_Y = [y for y in Y if not np.isnan(y)]
    func = interpolate.interp1d(valid_X,valid_Y,kind="cubic",bounds_error=False,fill_value="extrapolate")
    values_to_fill = func(inds_to_fill)
    for i,ind in enumerate(inds_to_fill):
        valid_Y.insert(ind,values_to_fill[i])
    return valid_Y

def preprocess(hist):
    X = [i for i in range(0,len(hist["Open"]))]
    # Clean initial data
    hist["Open"] = clean_data(X,hist["Open"])
    hist["Close"] = clean_data(X,hist["Close"])
    hist["High"] = clean_data(X,hist["High"])
    hist["Low"] = clean_data(X,hist["Low"])
    hist["Volume"] = clean_data(X,hist["Volume"])
    # Get the deltas for later params
    hist["dClose"] = delta(hist["Close"])
    # Get derived values
    hist["RSI"] = RSI(hist["dClose"],interval=14)
    hist["Momentum"] = momentum(hist["Close"],interval=10)
    hist["VolMom"] = momentum(hist["Volume"],interval=10)
    # Normalize the data
    hist["Open"] = normalize(hist["Open"])
    hist["Close"] = normalize(hist["Close"])
    hist["High"] = normalize(hist["High"])
    hist["Low"] = normalize(hist["Low"])
    hist["Volume"] = normalize(hist["Volume"])
    hist["Momentum"] = normalize(hist["Momentum"])
    hist["VolMom"] = normalize(hist["VolMom"])
    # get the normalized dClose
    hist["dOpen"] = delta(hist["Open"])
    hist["dClose"] = delta(hist["Close"])
    hist["dHigh"] = delta(hist["High"])
    hist["dLow"] = delta(hist["Low"])
    hist["dVolume"] = delta(hist["Volume"])
    hist["dMom"] = delta(hist["Momentum"])
    hist["dRSI"] = delta(hist["RSI"])

    # Set the index to X value
    hist["Index"] = X
    hist = hist.set_index('Index')
    # Delete params that aren't needed
    del hist["Dividends"]
    del hist["Stock Splits"]
    # del hist["Open"]
    # del hist["Close"]
    # del hist["High"]
    # del hist["Low"]
    return hist

def determine_quantity(stock,cash,percentage):
    stock_price = yahoo_price(stock,period='1d',interval='1m')[-1]
    if stock_price <= (cash * percentage):
        return "1"
    else:
        qty = round((percentage*cash)/stock_price,3)
        return str(qty)

def high_low_check(price_hist, period_length):
    """
    Check to see if the price history has reach a relative max or min in the last "period_length"
    number of periods. If so, send a "buy" or "sell" signal. If not, send a "pass" signal.
    """
    max_thres = 0.95
    min_thres = 0.05
    price_hist = np.asarray(price_hist)
    checkmax = np.where(price_hist[-period_length:] >= max_thres)
    checkmin = np.where(price_hist[-period_length:] <= min_thres)
    if np.any(checkmax):
        return -1 # if hitting max, sell
    elif np.any(checkmin):
        return 1 # if hitting min, buy
    else:
        return 0

def currentTime():
    now = datetime.datetime.utcnow()
    return datetime.datetime.strftime(now, "%m/%d/%Y %H:%M:%S")

