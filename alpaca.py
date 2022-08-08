from alpaca_trade_api.rest import REST, TimeFrame
import os

os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
api = REST('PKBOM4UMG57PJK849WPW', 'hj5hteWxT7IfNxPs4igq8xycl0T2F1ppU8zDlQgA', api_version='v2')
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

def buy(stock,qnty,order_type="market"):
    # print(f'Bought {qnty} shares of {stock}')
    api.submit_order(
        symbol=stock,
        qty=qnty,
        side='buy',
        type=order_type,
        time_in_force='day'
    )


def sell(stock,qnty,order_type="market"):
    # print(f'Sold {qnty} shares of {stock}')
    api.submit_order(
        symbol=stock,
        qty=qnty,
        side='sell',
        type=order_type,
        time_in_force='day'
    )


def nuclear():
    positions = get_positions()
    for position in positions:
        if position.side == "long":
            sell(position.symbol,float(position.qnty))
        else:
            buy(position.symbol, abs(float(position.qnty)))


def get_crypto_quote(sym):
    return api.get_crypto_snapshot(symbol=sym,exchange="CBSE").latest_quote.ap
