import datetime as DT
from stocks_db import get_stock_info, get_all_stock_info
from alpaca import check_market_hours, accountStatus, checkCash, get_positions
from functions import currentTime
from trade_methods import high_low_trade, SVR_trade, port_trade, position_pop

# Set tunable parameters
# Minimum allowed cash value before selling stocks to recoup losses
min_cash_limit = 800.0
# Minimum amount of time (in seconds) that much elapse before a stock can be bought/sold again.
min_wait_time = 72000.0
# Maximum percentage of total cash willing to risk on a single buy
max_risk = 0.05

# Get and state current time
print(f'{currentTime()}: Starting trade routine.')
# Check to see if market is open
if check_market_hours():
    print(f'{currentTime()} Market is open.')
    # Check for valid account status
    print(f'{currentTime()} Checking account status...')
    if accountStatus():
        print(f'{currentTime()} Account is valid.')
        # Check for sufficient funds
        print(f'{currentTime()} Checking cash status...')
        if checkCash() > min_cash_limit:
            print(f'{currentTime()} Sufficient cash.')
            # We are cleared to trade
            # First, get a list of all currently owned stocks/positions from Alpaca
            positions = get_positions()
            symbols = [position.symbol for position in positions]
            # Get a list of all stocks in our database that we want to trade
            stock_info = get_all_stock_info()[0:]
            stocks = [s[1] for s in stock_info]
            for stock in stocks:
                print(f'Processing stock: {stock}.')
                # Check to see if it's been at least [min_wait_time] hours since our last transaction with this stock
                last_transaction_time = DT.datetime.strptime(get_stock_info(stock)[3], "%m/%d/%Y %H:%M")
                time_delta = DT.datetime.utcnow() - last_transaction_time
                if time_delta.total_seconds() > min_wait_time:
                    print("\n")
                    print(f'{currentTime()}: Valid conditions for trading {stock}. Getting forecast.')
                    # Check if the stock is already in our portfolio
                    if stock in symbols:
                        # if so, check for any gains or big losses
                        action = port_trade(stock)
                        # If no gains or big losses, check for other trade opportunities
                        if action == 0:
                            # Do a min/max check first
                            signal = high_low_trade(stock,symbols,max_risk)
                            # If no trades made, do a SVR Forecast
                            if signal == 0:
                                SVR_trade(stock,max_risk)
                    else: # For stocks NOT currently in the portfolio
                        # Do a min/max check first
                        signal = high_low_trade(stock, symbols, max_risk)
                        # If no trades made, do a SVR Forecast
                        if signal == 0:
                            SVR_trade(stock, max_risk)
                else:
                    print(f'{currentTime()}: {stock} has been traded too recently. Passing for now.')
        else:
            # Close positions until funds are above the minimum
            print("\n")
            print(f'{currentTime()}: Not enough cash! Closing positions to raise funds...')
            cash = checkCash()
            while cash < min_cash_limit:
                position_pop(min_wait_time)
                cash = checkCash()
            print(f'{currentTime()}: Funds raised sufficiently. Normal trading will resume upon next run.')
    else:
        print(f'{currentTime()}: Account has been blocked! Check Alpaca for details.')
else:
    print(f'{currentTime()}: Market Closed. No trades made.')
print(f'{currentTime()}: COMPLETE.')











