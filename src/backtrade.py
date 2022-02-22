import datetime as DT
from functions import position_check
from svr import SVR_forecast_train as SVRT
from alpaca import get_positions

start_funds = 100000.0
stock = "AAPL"
data_length = 4464 # number of hours worth of data to collect


def create_date_list(start,end):
    current = start
    dates = []
    while current < end:
        dates.append(current)
        current = current + DT.timedelta(hours=24)
    return dates

# First, get a list of all currently owned stocks/positions
positions = get_positions()
symbols = [position.symbol for position in positions]
for stock in symbols:
    print("Checking stock "+stock)
    rec, reason = position_check(stock)
    print(rec, reason)

# # start by gettting a list of past dates to test on
# start = DT.datetime(2019,1,1)
# end = DT.datetime(2022,2,1)
# dates = create_date_list(start,end)
#
# position = None
# funds = start_funds
# transaction_price = 0.00
# final_price = 0.00
# print(f'Starting back test. Starting funds: ${funds}')
# for date in dates:
#     startdate = date - DT.timedelta(hours=4464)
#     print(f'Running Strategy for date: {DT.datetime.strftime(date,"%Y-%m-%d")}')
#     # get the data from yf
#     price_data = yahoo_old_price(stock,DT.datetime.strftime(startdate,"%Y-%m-%d"),DT.datetime.strftime(date,"%Y-%m-%d"))
#     # separate into forecast and verification data
#     forecast_prices = price_data[0:-1]
#     final_price = price_data[-1]
#     # Get the forecast
#     rec, reason = SVRT(forecast_prices)
#     print(f'Recommendation: {rec}; Reason: {reason}')
#     if rec == "BUY" and position == None:
#         # Open a long position:
#         transaction_price = forecast_prices[-1]
#         position = "LONG"
#         print(f'Opening a LONG position at price ${transaction_price}.')
#     elif rec == "BUY" and position == "SHORT":
#         # close a short position
#         net = transaction_price - final_price
#         funds = funds + net
#         position = None
#         print(f'Closing a SHORT position at price ${final_price}.')
#         print(f'Net gain/loss: ${net}')
#         print(f'Funds are now ${funds}.')
#     elif rec == "SELL" and position == None:
#         # open a short position
#         transaction_price = forecast_prices[-1]
#         position = "SHORT"
#         print(f'Opening a SHORT position at price ${transaction_price}.')
#     elif rec == "SELL" and position == "LONG":
#         # Close a long position
#         net = final_price - transaction_price
#         funds = funds + net
#         position = None
#         print(f'Closing a LONG position at price ${final_price}.')
#         print(f'Net gain/loss: ${net}')
#         print(f'Funds are now ${funds}.')
#     else:
#         print(f'Passed for now.')
#
# # Close any open positions at the end
# if position == "LONG":
#     funds = funds + (final_price - transaction_price)
# elif position == "SHORT":
#     funds = funds + (transaction_price - final_price)
#
# print(f'Ending funds: ${funds}')
# print(f'Total gain/loss: ${funds - start_funds}')



