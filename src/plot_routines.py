import matplotlib.pyplot as plt
import numpy as np
from functions import preprocess, yahoo_hist, SMA, bbands, EMA, RSI

def plot_scatter(X, Y, xlabel, ylabel, title=""):
    plt.figure()
    plt.scatter(X, Y, marker='o', color='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.5)
    plt.show()

def plot_line(X, Y, xlabel, ylabel, title=""):
    plt.figure()
    plt.plot(X, Y, color='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.5)
    plt.show()

def plot_histogram(Y):
    plt.figure()
    # plt.hist(Y,bins=np.arange(-10.0,10.0,0.5))
    plt.hist(Y)
    plt.show()

def candle_chart(stock,df):
    width = 0.01
    stick = 0.001
    up = df[df["Close"]>=df["Open"]]
    down = df[df["Close"]<df["Open"]]
    # define colors
    up_color = "g"
    down_color = "r"

    plt.figure()
    # Plot up prices
    plt.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color=up_color, edgecolor="None")
    plt.bar(up.index, up.High - up.Close, stick, bottom=up.Close, color=up_color, edgecolor="None")
    plt.bar(up.index, up.Low - up.Open, stick, bottom=up.Open, color=up_color, edgecolor="None")
    # plot down prices
    plt.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color=down_color, edgecolor="None")
    plt.bar(down.index, down.High - down.Open, stick, bottom=down.Open, color=down_color, edgecolor="None")
    plt.bar(down.index, down.Low - down.Close, stick, bottom=down.Close, color=down_color, edgecolor="None")

    # rotate x-axis tick labels
    plt.xticks(rotation=45, ha='right')
    # Add title and labels
    plt.title(f'{stock} Technical')
    plt.xlabel("Date/Time")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig("candle_chart_"+stock+".png")

def tech_chart(stock,df):
    plt.figure()
    temp = df.loc[:, df.columns != 'Volume']
    mean_line = temp.mean(axis=1)
    sma5 = SMA(df,5)
    sma20 = SMA(df,20)
    bb = bbands(df)
    plt.plot_date(df.index,df['High'], marker="", linestyle='-',color='b', alpha=0.3)
    plt.plot_date(df.index, df['Low'], marker="", linestyle='-', color='b', alpha=0.3)
    plt.plot_date(df.index, df['Open'], marker="", linestyle='-', color='g', alpha=0.3)
    plt.plot_date(df.index, df['Close'], marker="", linestyle='-', color='g', alpha=0.3)
    plt.plot_date(df.index, mean_line, marker="", linestyle='-', color='k', alpha=0.9)
    plt.plot_date(df.index, sma5, marker="", linestyle='-', color='r', alpha=0.9)
    plt.plot_date(df.index, sma20, marker="", linestyle='--', color='r', alpha=0.9)
    plt.plot_date(df.index, bb["BB_UPPER"], marker="", linestyle='-', color='y', alpha=0.9)
    plt.plot_date(df.index, bb["BB_LOWER"], marker="", linestyle='-', color='y', alpha=0.9)

    # rotate x-axis tick labels
    plt.xticks(rotation=45, ha='right')
    plt.xlim(df.index[0],df.index[-1])
    # Add title and labels
    plt.title(f'{stock} Technical')
    plt.xlabel("Date/Time")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig("tech_chart_"+stock+".png")

# stock = "AMZN"
# yahoo_stock_info(stock)
# hist = yahoo_hist(stock,period="3mo",interval="1d")
# hist = preprocess(hist)



