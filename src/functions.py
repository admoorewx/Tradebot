from scipy import interpolate
from scipy import stats
from scipy.stats import spearmanr
from finta import TA

import yfinance as YF
import numpy as np
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
    hist = preprocess(hist)
    return hist

def yahoo_price(stock,period="1y",interval="1d"):
    """
    Returns just the closing stock price of the input stock ticker.
    """
    df = yahoo_hist(stock,period,interval)
    return df["Close"]

def yahoo_current_price(stock):
    stk = YF.Ticker(stock)
    return stk.info["currentPrice"]

def clean_data(X,Y):
    """
    Use scipy interpolate 1D to perform a linear interpolation to fill NaN values.
    Note: this will extrapolate if NaNs are found at the start or end of "Y".
    """
    inds_to_fill = np.where(np.isnan(Y))[0]
    valid_X = [x for i,x in enumerate(X) if not np.isnan(Y[i])]
    valid_Y = [y for y in Y if not np.isnan(y)]
    func = interpolate.interp1d(valid_X,valid_Y,kind="linear",bounds_error=False,fill_value="extrapolate")
    values_to_fill = func(inds_to_fill)
    for i,ind in enumerate(inds_to_fill):
        valid_Y.insert(ind,values_to_fill[i])
    return valid_Y

def preprocess(df):
    # Create a new dataframe to return
    new_df = pd.DataFrame()
    # Figure out the common interval between time stamps
    timestamps = [i for i in df.index]
    deltas = []
    for t,time in enumerate(timestamps):
        try:
            delta = time - timestamps[t-1]
            deltas.append(delta.total_seconds())
        except:
            pass
    mode = stats.mode(deltas)[0][0]
    # Now create a new time stamp series with timestamps at each interval (i.e. no gaps)
    start = timestamps[0]
    end = timestamps[-1]
    current = start
    new_ts = []
    while current <= end:
        new_ts.append(current)
        current = current + datetime.timedelta(seconds=mode)
    # Set the new index
    new_df.index = new_ts
    # Create a series of X values for interpolation in clean_data()
    X = [i for i in range(0,len(new_ts))]
    # Figure out where the original data fits within the new_ts series
    inds = []
    for i in timestamps:
        try:
            inds.append(new_ts.index(i))
        except:
            pass
    # Loop through each column in the df and fill in missing data
    # Assign the filled-in data to new_df
    for item in df:
        Y = [np.nan for i in range(0,len(new_ts))]
        for x in range(0,len(inds)):
            Y[inds[x]] = df[item][x]
        new_df[item] = clean_data(X,Y)
    # Return the new dataframe
    return new_df

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

def SMA(df,window):
    return TA.SMA(df,window)

def bbands(df):
    return TA.BBANDS(df)

def EMA(df,window):
    return TA.EMA(df,window)

def RSI(df):
    return TA.RSI(df)

def VWAP(df):
    return TA.VWAP(df)

def markmo(df):
    return TA.MOM(df)