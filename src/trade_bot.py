#import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import REST, TimeFrame
from scipy import interpolate
from scipy.optimize import curve_fit, leastsq
import datetime as DT
import yfinance as YF
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
api = REST('PK9KGYIPW7688M5ID39S', 'CrnWvypaW0iCZHqGgL2O3QjJbBPIVAImBJFeVcBX', api_version='v2')
account = api.get_account()

def accountStatus(account):
    if account.trading_blocked:
        print("Account is currently blocked.")
        return False
    else:
        return True

def checkBalance(account):
    return account.buying_power

def gainloss(account):
    return float(account.equity) - float(account.last_equity)

def yahoo_hist_day(stock):
    dat = YF.Ticker(stock)
    hist = dat.history(period="1d",interval="1m")
    return hist

def yahoo_hist_year(stock):
    dat = YF.Ticker(stock)
    hist = dat.history(period="1y",interval="1wk")
    return hist

def yahoo_hist_month(stock):
    dat = YF.Ticker(stock)
    hist = dat.history(period="31d",interval="1d")
    return hist

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

def forecast(X,Y,deg=1,fcst_range=10):
    coeff = np.polynomial.polynomial.polyfit(X,Y,deg=deg)
    fcst = []
    for x in range(0,fcst_range):
        x = X[-1] + x
        y = 0.0
        for i in range(0,deg):
            if i == 0:
                y = y + coeff[i]
            else:
                y = y + coeff[i] * (x**i)
        fcst.append(y)
    return fcst

def cos_func(x,a,b,phi,c):
    return a * np.cos(b*x+phi) + c

def sin_func(x,a,b,phi,c):
    return a * np.sin(b*x+phi) + c

def sin_func2(x,a1,b1,p1,a2,b2,p2,c):
    return a1*np.sin(b1*x+p1) + a2*np.sin(b2*x+p2) + c

def sin_lin_func(x,a,b,c,d,e):
    return a + b*x + c*np.sin(d*x+e)

def running_average(interval,window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval,window,'same')

def normalize(x):
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    return [(2*(X-mn)/(mx-mn)) -1 for X in x]

def slope_est(Y,X,window=5):
    num = np.nanmean(Y[0:window]) - np.nanmean(Y[-window:-1])
    denom = np.nanmean(X[0:window]) - np.nanmean(X[-window:-1])
    return num/denom

def root_mean_squared_error(truth,est):
    truth = np.asarray(truth)
    est = np.asarray(est)
    if len(truth) == len(est):
        return np.sqrt((np.sum((truth-est))**2.0)/len(truth))
    else:
        print(f'ERROR: input arrays are not the same length! ({len(truth)} and {len(est)})')
        return None

def func_fit(X,Y):
    # smooth = 10
    # run_avg = running_average(Y,smooth)
    # run_avg = run_avg[1:-2] # Chop off the first and last values since they're errantly low from the running average
    # X = X[1:-2] # make X the same length
    # normed = normalize(run_avg)
    # guess_slope = slope_est(Y,X)
    # normed = normalize(ffit)
    # guess_freq = zero_freq(normed)
    # guess_amp = np.std(ffit) * 2.**0.5
    # guess_phase = 0.0
    # guess_offset = np.nanmean(ffit)
    # p0 = [guess_slope, guess_freq, guess_amp, guess_phase, guess_offset]
    # coeff = curve_fit(sin_lin_func, X, ffit)[0]
    # fit = [sin_lin_func(x, coeff[0], coeff[1], coeff[2], coeff[3], coeff[4]) for x in X]
    normed = normalize(Y)
    # ffit = fft(normed)
    guess_freq = fft(normed)
    guess_amp = np.std(Y) * 2. ** 0.5
    guess_phase = 0.0
    guess_offset = np.nanmean(Y)
    p0 = [guess_freq, guess_amp, guess_phase, guess_offset]
    coeff = curve_fit(cos_func,X,Y,p0=p0)[0]
    fit = [sin_func(x, coeff[0], coeff[1], coeff[2], coeff[3]) for x in X]
    rmse = root_mean_squared_error(Y,fit)
    return fit, rmse

def best_fit(X,Y):
    windows = np.arange(5,15)
    best_window = windows[0]
    fit, best_rmse = func_fit(X,Y,smooth=windows[0])
    for window in windows[1:]:
        new_fit, new_rmse = func_fit(X,Y,smooth=window)
        print(new_rmse)
        if new_rmse < best_rmse:
            best_window = window
            fit = new_fit
            best_rmse = new_rmse
    print(f'The best fit used window size: {best_window}, resulting in an RMSE value of {best_rmse}.')
    return fit

def fft(Y):
    n = len(Y)
    freq_filter = 3.0
    fhat = np.fft.fft(Y,n)
    PSD = fhat * np.conj(fhat) / n
    freq = (1.0/(n)) * np.arange(n)
    L = np.arange(1,np.floor(n/2.0),dtype='int')

    plt.figure()
    plt.plot(freq[L],PSD[L])
    plt.savefig("/home/icebear/Tradebot/psd.png")
    print(L)
    print(list(PSD[L]).index(max(PSD[L])))
    ind = list(PSD[L]).index(max(PSD[L]))
    return freq[ind]

    # filter = int(max(PSD[L])/freq_filter)
    # indicies = PSD > filter
    # PSDclean = PSD * indicies
    # fhat = indicies * fhat
    # ffilt = np.fft.ifft(fhat)
    # x = [i for i in range(0,len(Y))]
    #
    # # plt.figure()
    # # plt.plot(x,ffilt)
    # # plt.savefig("/home/icebear/Tradebot/ffilt.png")
    # ffilt = [x.real for x in ffilt]
    # return ffilt

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

def rel_freq(x):
    x = [round(i) for i in x]
    freq = [list(x).count(value) / len(x) for value in x]
    return max(freq)

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    #guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_freq = rel_freq(yy)
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy[0:int(len(yy)/4.0)]) # getting first guess offset from first 1/2 of data
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    popt = curve_fit(sin_func2, tt, yy, p0=guess)[0]
    a1, b1, p1, a2, b2, p2, c = popt
    return [a1, b1, p1, a2, b2, p2, c]

def retrieve_variables(hist,set="Open"):
    X = [i for i in range(0,len(hist["Open"]))]
    try:
        Y = [y for y in hist[set]]
    except:
        print("ERROR: Invalid data type from stock history.")
    return X,Y

def plot_history(hists):
    plt.figure()
    colors = ['r','g','b']
    for i, hist in enumerate(hists):
        dates = hist.index
        open = hist["Open"]
        close = hist["Close"]
        plt.plot_date(dates,open,color=colors[i])
        # plt.plot_date(dates,close,color='r')
    plt.savefig("/home/icebear/Tradebot/history.png")

def plot_history_fcst(X,hist,fcst,fit):
    X_fcst = [X[-1] + i for i in range(1,len(fcst)+1)]
    plt.figure()
    plt.plot(X,hist,color='r',label="History")
    plt.plot(X_fcst,fcst,color='g',label="Forecast")
    plt.plot(X, fit, color='b', label="Forecast")
    plt.xlim(0,X_fcst[-1])
    #plt.xlim(0,50)
    plt.ylim(110,300)
    plt.savefig("/home/icebear/Tradebot/forecast.png")

def plot_fit(X,hist,fit):
    plt.figure()
    plt.plot(X,hist,color='r',label="History")
    plt.plot(X, fit, color='b', label="Forecast")
    plt.ylim(np.nanmin(hist),np.nanmax(hist))
    plt.savefig("/home/icebear/Tradebot/fit.png")

def plot_norm(X,fit):
    if len(fit) != len(X):
        center = len(X) - len(fit)
        fit_x = X[center:]
    else:
        fit_x = X
    plt.figure()
    plt.plot(fit_x, fit, color='b', label="Forecast")
    plt.ylim(-1,1)
    plt.savefig("/home/icebear/Tradebot/normed.png")

accountStatus(account)
day = yahoo_hist_day("AAPL")
# year = yahoo_hist_year("AAPL")
# month = yahoo_hist_month("AAPL")
# plot_history([year])

X,Y = retrieve_variables(day,set="Open")
Y = clean_data(X,Y)
fit,rmse = func_fit(X,Y)
plot_fit(X,Y,fit)

#fit, rmse = func_fit(X,Y,smooth=100)


print("Success.")
