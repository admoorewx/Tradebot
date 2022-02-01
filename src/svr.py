import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, NuSVR
from functions import normalize, root_mean_squared_error, yahoo_price, clean_data

def SVR_forecast(stock,period="6mo",interval="1h",forecast_length=48,ensemble_members=100):
    """
    Performs an SVR regression ensemble forecast and returns a recommendation
    stock = target stock ticker (string)
    period = length of time to get price data over (string) examples: "1mo", "3mo", "1w", "1y"...
    interval = frequency of price data (string) examples: "1m", "5m", "1h", "1d"...
    forecast_length = Number of hours to produce a forecast for (int)
    ensemble_members = The number of ensemble members to use in the forecast (int)
    Returns: String specifying a buy, sell, or hold recommendation.
    """

    # # Get the data from the YF API
    price = yahoo_price(stock,period=period,interval=interval)
    # Clean the data and normalize
    price = normalize(clean_data(np.arange(0,len(price)),price))

    X = np.arange(0,len(price))
    y = np.asarray(price)

    forecast_periods = [X[-1] + i for i in range(0,forecast_length)]
    forecast_periods = np.asarray(forecast_periods)
    forecast_window_start = -4 # default is # of hours before the default 24-hour mark.
    # Recommendation is to keep this value at 5 hours or less to help avoid PDT rules and give a more accurate forecast.

    recs = []
    valid_forecasts = []
    model_fits = []
    for i in range(0,ensemble_members):
        # Split the training/testing data
        X1, X2, y1, y2 = train_test_split(X, y,train_size=0.7)
        # Create the SVR model
        model = NuSVR(kernel='rbf', tol=np.std(y),C=np.std(y))
        # Fit the data
        model.fit(X1.reshape(-1,1), y1)
        # Collect model fit statistics
        ypredict = model.predict(X2.reshape(-1,1))
        model_fit_error = root_mean_squared_error(y[X2],ypredict)
        model_fits.append(model_fit_error)
        # Create the forecast and get error statistics
        forecast = model.predict(forecast_periods.reshape(-1,1))
        forecast_errors = [np.random.normal(loc=0.0,scale=(np.std(y)*(i/(5.0*len(forecast))))) for i in range(len(forecast))]
        forecast = forecast + forecast_errors
        # Find the RMSE of the forecast

        # Find the last model fitted price value before the forecast began
        last_ind = np.where(X2 == np.nanmax(X2))[0]
        last_yp = ypredict[last_ind]

        # See if the first forecast value is substantially difference than the last known price value
        # If so, don't use this forecast.
        if np.abs(forecast[0]-y[-1]) > (np.std(y) * 0.5):
            valid_forecasts.append(-1)
        else: # Only get forecasts from the valid ensemble members.
            valid_forecasts.append(1)
            for ff in forecast[forecast_window_start:]:
                recs.append((ff-last_yp))

    # Get the buy/sell/hold recommendation and reason
    if np.mean(valid_forecasts) > 0.4:
        if np.mean(recs) > 0.0:
            reccomendation = "BUY"
            reason = "Price Expected to rise."
        elif np.mean(recs) < 0.0:
            reccomendation =  "SELL"
            reason = "Price Expected to fall."
        else:
            reccomendation =  "PASS"
            reason = "Little price fluctuation expected."
    elif np.var(model_fits) <= 0.008:
        reccomendation =  "PASS"
        reason = "Potential model overfit."
    else:
        reccomendation = "PASS"
        reason = "Inaccurate forecast initialization."
    return reccomendation, reason


def SVR_forecast_train(stock,period="6mo",interval="1h",forecast_length=24,ensemble_members=100):
    """
    Performs an SVR regression ensemble forecast and returns a recommendation
    stock = target stock ticker (string)
    period = length of time to get price data over (string) examples: "1mo", "3mo", "1w", "1y"...
    interval = frequency of price data (string) examples: "1m", "5m", "1h", "1d"...
    forecast_length = Number of hours to produce a forecast for (int)
    ensemble_members = The number of ensemble members to use in the forecast (int)
    Returns: String specifying a buy, sell, or hold recommendation.
    """

    # # Get the data from the YF API
    price = yahoo_price(stock,period=period,interval=interval)
    # Clean the data and normalize
    price = normalize(clean_data(np.arange(0,len(price)),price))

    X = np.arange(0,len(price))
    y = np.asarray(price)

    start = 0
    end = len(y) - forecast_length
    min_hours = 19

    recs = []
    valid_forecasts = []
    error = []
    model_fits = []
    for i in range(0,ensemble_members):
        # Split the training/testing data
        X1, X2, y1, y2 = train_test_split(X[start:end], y[start:end],train_size=0.7)
        # Create the SVR model
        model = NuSVR(kernel='rbf', tol=np.std(y))
        # Fit the data
        model.fit(X1.reshape(-1,1), y1)
        # Collect model fit statistics
        ypredict = model.predict(X2.reshape(-1,1))
        model_fit_error = root_mean_squared_error(y[X2],ypredict)
        model_fits.append(model_fit_error)
        # Create the forecast and get error statistics
        forecast = model.predict(X[end:].reshape(-1,1))
        forecast_errors = [np.random.normal(loc=0.0,scale=(np.std(y)*(i/(5.0*len(forecast))))) for i in range(len(forecast))]
        forecast = forecast + forecast_errors
        # Find the RMSE of the forecast
        score = root_mean_squared_error(y[end:],forecast)
        error.append(score)
        # Find the last model fitted price value before the forecast began
        last_ind = np.where(X2 == np.nanmax(X2))[0]
        last_yp = ypredict[last_ind]
        window = len(forecast) - min_hours

        # See if the first forecast value is substantially difference than the last known price value
        # If so, don't use this forecast.
        if np.abs(forecast[0]-y[end]) > (np.std(y) * 0.5):
            valid_forecasts.append(-1)
        else: # Only get forecasts from the valid ensemble members.
            valid_forecasts.append(1)
            for ff in forecast[window:]:
                recs.append((ff-last_yp))

    # Get the buy/sell/hold recommendation
    if np.mean(valid_forecasts) > 0.4:
        if np.mean(recs) > 0.0:
            recommend = "BUY"
            reason = "Price Expected to rise."
            net = y[-1] - y[end]
        elif np.mean(recs) < 0.0:
            recommend = "SELL"
            reason = "Price Expected to fall."
            net = y[-1] - y[end]
        else:
            recommend = "PASS"
            reason = "Little price fluctuation expected."
            net = 0.0
    elif np.var(model_fits) <= 0.008:
        recommend = "PASS"
        reason = "Potential model overfit."
        net = 0.0
    else:
        recommend = "PASS"
        reason = "Inaccurate model initialization."
        net = 0.0

    return recommend, reason, net


def SVR_forecast_plot(stock,period="6mo",interval="1h",forecast_length=48,ensemble_members=100):
    """
    --- Same as SVR_forecast, but creates a plot ---
    Performs an SVR regression ensemble forecast and returns a recommendation
    stock = target stock ticker (string)
    period = length of time to get price data over (string) examples: "1mo", "3mo", "1w", "1y"...
    interval = frequency of price data (string) examples: "1m", "5m", "1h", "1d"...
    forecast_length = Number of hours to produce a forecast for (int)
    ensemble_members = The number of ensemble members to use in the forecast (int)
    Returns: String specifying a buy, sell, or hold recommendation.
    """

    # # Get the data from the YF API
    price = yahoo_price(stock,period=period,interval=interval)
    # Clean the data and normalize
    price = normalize(clean_data(np.arange(0,len(price)),price))

    X = np.arange(0,len(price))
    y = np.asarray(price)

    forecast_periods = [X[-1] + i for i in range(0,forecast_length)]
    forecast_periods = np.asarray(forecast_periods)
    forecast_window_start = -4 # default is # of hours before the default 24-hour mark.
    # Recommendation is to keep this value at 5 hours or less to help avoid PDT rules and give a more accurate forecast.

    recs = []
    valid_forecasts = []
    model_fits = []
    plt.figure()
    for i in range(0,ensemble_members):
        # Split the training/testing data
        X1, X2, y1, y2 = train_test_split(X, y,train_size=0.7)
        # Create the SVR model
        model = NuSVR(kernel='rbf', tol=np.std(y),C=np.std(y))
        # Fit the data
        model.fit(X1.reshape(-1,1), y1)
        # Collect model fit statistics
        ypredict = model.predict(X2.reshape(-1,1))
        model_fit_error = root_mean_squared_error(y[X2],ypredict)
        model_fits.append(model_fit_error)
        # Create the forecast and get error statistics
        forecast = model.predict(forecast_periods.reshape(-1,1))
        forecast_errors = [np.random.normal(loc=0.0,scale=(np.std(y)*(i/(5.0*len(forecast))))) for i in range(len(forecast))]
        forecast = forecast + forecast_errors
        # Find the RMSE of the forecast

        # Find the last model fitted price value before the forecast began
        last_ind = np.where(X2 == np.nanmax(X2))[0]
        last_yp = ypredict[last_ind]

        # See if the first forecast value is substantially difference than the last known price value
        # If so, don't use this forecast.
        if np.abs(forecast[0]-y[-1]) > (np.std(y) * 0.5):
            valid_forecasts.append(-1)
            color = 'b'
        else: # Only get forecasts from the valid ensemble members.
            valid_forecasts.append(1)
            color = 'g'
            for ff in forecast[forecast_window_start:]:
                recs.append((ff-last_yp))

        # Plot the model fit and the forecast
        plt.scatter(X2,ypredict,color='r', alpha=0.2)
        plt.plot(forecast_periods,forecast,color=color,alpha=0.2)

    # Get the buy/sell/hold recommendation and reason
    if np.mean(valid_forecasts) > 0.4:
        if np.mean(recs) > 0.0:
            reccomendation = "BUY"
            reason = "Price Expected to rise."
        elif np.mean(recs) < 0.0:
            reccomendation =  "SELL"
            reason = "Price Expected to fall."
        else:
            reccomendation =  "PASS"
            reason = "Little price fluctuation expected."
    elif np.var(model_fits) <= 0.008:
        reccomendation =  "PASS"
        reason = "Potential model overfit."
    else:
        reccomendation = "PASS"
        reason = "Inaccurate forecast initialization."

    # Plot the price data and save the figure
    plt.plot(X,y,color='k')
    plt.title(f'{stock} Forecast - Recommendation: {reccomendation}')
    plt.xlim(0,len(X)+len(forecast))
    plt.ylim(0,1.2)
    plt.savefig(stock+".png")
    plt.close('all')

    return reccomendation, reason




def SVR_forecast_train_diagnosis(stock,period="6mo",interval="1h",forecast_length=48,ensemble_members=100):
    """
    Performs an SVR regression ensemble forecast and returns a recommendation. Main difference from
    the traditional SVR_forecast is that this version saves figures and outputs helpful statistics.
    stock = target stock ticker (string)
    period = length of time to get price data over (string) examples: "1mo", "3mo", "1w", "1y"...
    interval = frequency of price data (string) examples: "1m", "5m", "1h", "1d"...
    forecast_length = Number of hours to produce a forecast for (int)
    ensemble_members = The number of ensemble members to use in the forecast (int)
    Returns: String specifying a buy, sell, or hold recommendation.
    """

    # # Get the data from the YF API
    price = yahoo_price(stock,period=period,interval=interval)
    # Clean the data and normalize
    price = normalize(clean_data(np.arange(0,len(price)),price))

    X = np.arange(0,len(price))
    y = np.asarray(price)

    start = 0
    end = len(y) - forecast_length
    min_hours = 20

    recs = []
    valid_forecasts = []
    error = []
    model_fits = []
    plt.figure()
    for i in range(0,ensemble_members):
        # Split the training/testing data
        X1, X2, y1, y2 = train_test_split(X[start:end], y[start:end],train_size=0.7)
        # Create the SVR model
        #model = SVR(kernel='rbf',tol=np.std(y), epsilon=np.var(y))
        model = NuSVR(kernel='rbf', tol=np.std(y))
        # Fit the data
        model.fit(X1.reshape(-1,1), y1)
        # Collect model fit statistics
        ypredict = model.predict(X2.reshape(-1,1))
        model_fit_error = root_mean_squared_error(y[X2],ypredict)
        model_fits.append(model_fit_error)
        # Create the forecast and get error statistics
        forecast = model.predict(X[end:].reshape(-1,1))
        forecast_errors = [np.random.normal(loc=0.0,scale=(np.std(y)*(i/(5.0*len(forecast))))) for i in range(len(forecast))]
        forecast = forecast + forecast_errors
        score = root_mean_squared_error(y[end:],forecast)
        error.append(score)
        # Find the last model fitted price value before the forecast began
        last_ind = np.where(X2 == np.nanmax(X2))[0]
        last_yp = ypredict[last_ind]
        window = len(forecast) - min_hours

        # See if the first forecast value is substantially difference than the last known price value
        # If so, don't use this forecast.
        if np.abs(forecast[0]-y[end]) > (np.std(y) * 0.5):
            color='b'
            valid_forecasts.append(-1)
        else: # Only get forecasts from the valid ensemble members.
            color='g'
            valid_forecasts.append(1)
            for ff in forecast[window:]:
                recs.append((ff-last_yp))

        # Plot the model fit and the forecast
        plt.scatter(X2,ypredict,color='r', alpha=0.2)
        plt.plot(X[end:],forecast,color=color,alpha=0.2)

    # Get the buy/sell/hold recommendation
    if np.mean(valid_forecasts) > 0.4:
        if np.mean(recs) > 0.0:
            recommend = "BUY"
            reason = "Price Expected to rise."
            net = y[-1] - y[end]
        elif np.mean(recs) < 0.0:
            recommend = "SELL"
            reason = "Price Expected to fall."
            net = y[end] - y[-1]
        else:
            recommend = "PASS"
            reason = "Little price fluctuation expected."
            net = 0.0
    elif np.var(model_fits) <= 0.008:
        recommend = "PASS"
        reason = "Potential model overfit."
        net = 0.0
    else:
        recommend = "PASS"
        reason = "Inaccurate model initialization."
        net = 0.0

    # Plot the price data and save the figure
    plt.plot(X,y,color='k')
    plt.title(f'{stock} Forecast - Recommendation: {recommend}')
    plt.xlim(0,len(X))
    plt.ylim(0,1.2)
    plt.savefig(stock+".png")
    plt.close('all')

    # Print out some statistics
    # print(f'Produced {len(error)} forecasts for {stock}:')
    # print(f'Forecast average RMSE: {np.mean(error)}')
    # print(f'Forecast Std Deviation: {np.std(error)}')
    # print(f'Forecast Variance: {np.var(error)}')
    # print(f'Model Fit Error Variance: {np.var(model_fits)}')
    # print(f'Model Fit Error Std Deviation: {np.std(model_fits)}')
    # print(f'{stock} Std Deviation: {np.std(y)}')
    # print(f'{stock} Variance: {np.var(y)}')
    # print(f'Mean Rec Value: {np.mean(recs)}')
    # print(f'Recommendation: {recommend}\n')
    # Return the recommendation (and for now the net change)
    return recommend, reason, net