from alpaca_trade_api.rest import REST, TimeFrame
import os

os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
api = REST('', '', api_version='v2')
account = api.get_account()


def accountStatus():
    if account.trading_blocked:
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
        return True
    else:
        return False

def tradable_status(stock):
    asset = api.get_asset(stock)
    if asset.tradable:
        return True
    else:
        return False

def get_positions():
    return api.list_positions()

def get_position(stock):
    return api.get_position(stock)

def get_position_quantity(stock):
    return api.get_position(stock).qty

def all_avail_stocks():
    return api.list_assets(status='active')

def buy(stock,qnty):
    # print(f'Bought {qnty} shares of {stock}')
    api.submit_order(
        symbol=stock,
        qty=qnty,
        side='buy',
        type='market',
        time_in_force='day'
    )


def sell(stock,qnty):
    # print(f'Sold {qnty} shares of {stock}')
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
        if position.side == "long":
            sell(position.symbol,float(position.qnty))
        else:
            buy(position.symbol, abs(float(position.qnty)))
