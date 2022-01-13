from alpaca_trade_api.rest import REST, TimeFrame
import os

os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
api = REST('', '', api_version='v2')
account = api.get_account()


def accountStatus():
    if account.trading_blocked:
        print("Account is currently blocked.")
        return False
    else:
        return True

def checkBalance():
    return float(account.buying_power)

def checkCash():
    return float(account.cash)

def checkEquity():
    return float(account.equity)

def checkPDT():
    return account.patter_day_trader

def gainloss():
    return float(account.equity) - float(account.last_equity)

def check_market_hours():
    clock = api.get_clock()
    if clock.is_open:
        print("The market is open!")
        return True
    else:
        print("The market is closed!")
        return False

def tradable_status(stock):
    asset = api.get_asset(stock)
    if asset.tradable:
        return True
    else:
        return False

def get_positions():
    return api.list_positions()

def get_position_quantity(stock):
    return api.get_position(stock).qty

def buy(stock,qnty):
    api.submit_order(
        symbol=stock,
        qty=qnty,
        side='buy',
        type='market',
        time_in_force='day'
    )

def sell(stock,qnty):
    api.submit_order(
        symbol=stock,
        qty=qnty,
        side='sell',
        type='market',
        time_in_force='day'
    )

def nuclear():
    positions = get_positions()
    for position in positions:
        sell(position.symbol,position.qnty)
