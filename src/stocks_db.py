import sqlite3
import json
import datetime
from alpaca import checkCash, checkEquity, gainloss
from emailer import send_email

def currentTime():
    now = datetime.datetime.utcnow()
    return datetime.datetime.strftime(now, "%m/%d/%Y %H:%M:%S")

def database():
    connection = sqlite3.connect('stocks.db')
    cursor = connection.cursor()
    return connection, cursor

def initialize():
    """
    This function will initialize the SQLite database to hold
    stock transaction information. The database will have the following formatting:
    stock_number: integer, stock ID number for the database
    symbol: string, stock symbol as it appears in most trading platforms.
    position: string, describes current position (either "LONG", "SHORT", or "NONE".
    last_transaction: string, either "buy", "sell" or "passed".
    last_transaction_time: string, datetime string of last buy/sell transaction.
    bought_price: real (float), the stock price at the time of purchase.
    sold_price: real (float), the stock price at the time of sale.
    net: real (float), the net gain/loss from a long or short position.
    transaction_reason: string, reason why the last transaction was executed.
    """
    create_talbes = """CREATE TABLE stocks (
    stock_number INTEGER PRIMARY KEY,
    symbol TEXT,
    position TEXT,
    last_transaction TEXT,
    last_transaction_time TEXT,
    bought_price REAL,
    sold_price REAL,
    net REAL,
    transaction_reason TEXT
    )"""
    connection, cursor = database()
    cursor.execute(create_talbes)
    connection.commit()
    connection.close()
    print(f'{currentTime()}: Stock database created successfully')

def add_stock_from_json(json_file):
    connection, cursor = database()
    with open(json_file) as json_file:
        stocks = json.load(json_file).keys()
        for i,stock in enumerate(stocks):
            add_command = f'INSERT INTO stocks VALUES("{i}","{stock}","NONE","Pass","01/01/2022 00:00","0.00","0.00","0.00","initial");'
            cursor.execute(add_command)
    connection.commit()
    connection.close()
    print(f'{currentTime()}: Added stocks successfully.')

def get_all_stock_info():
    connection, cursor = database()
    cursor.execute("SELECT * FROM stocks")
    stocks = cursor.fetchall()
    connection.close()
    return stocks

def get_stock_info(stock):
    connection, cursor = database()
    cursor.execute(f'SELECT symbol, position, last_transaction, last_transaction_time, bought_price, sold_price, net, transaction_reason FROM stocks WHERE symbol = "{stock}";')
    response = cursor.fetchone()
    connection.close()
    return response

def update_stock_transaction(stock,transaction,transaction_time,reason):
    connection, cursor = database()
    cursor.execute(f'UPDATE stocks SET last_transaction = "{transaction}" WHERE symbol="{stock}";')
    cursor.execute(f'UPDATE stocks SET last_transaction_time = "{transaction_time}" WHERE symbol="{stock}";')
    cursor.execute(f'UPDATE stocks SET transaction_reason = "{reason}" WHERE symbol="{stock}";')
    connection.commit()
    connection.close()
    print(f'{currentTime()}: Updated stock {stock} transaction successfully.')

def update_position(stock,position):
    connection, cursor = database()
    cursor.execute(f'UPDATE stocks SET position= "{position}" WHERE symbol="{stock}";')
    connection.commit()
    connection.close()

def update_bought_price(stock,price,owned):
    price = round(price,2)
    connection, cursor = database()
    if owned: # Exiting a short position
        update_position(stock, "NONE")
        cursor.execute(f'SELECT sold_price FROM stocks WHERE symbol = "{stock}";')
        sp = cursor.fetchone()[0]
        net = sp - price
        cursor.execute(f'UPDATE stocks SET net = "{net}" WHERE symbol="{stock}";')
    else: # entering a long position
        update_position(stock, "LONG")
        cursor.execute(f'UPDATE stocks SET net = "0.00" WHERE symbol="{stock}";')
        cursor.execute(f'UPDATE stocks SET sold_price = "0.00" WHERE symbol="{stock}";')
    cursor.execute(f'UPDATE stocks SET bought_price = "{price}" WHERE symbol="{stock}";')
    connection.commit()
    connection.close()
    print(f'{currentTime()}: Updated stock {stock} prices successfully.')


def update_sold_price(stock,price,owned):
    price = round(price, 2)
    connection, cursor = database()
    if owned: # Exiting a long position
        update_position(stock,"NONE")
        cursor.execute(f'SELECT bought_price FROM stocks WHERE symbol = "{stock}";')
        bp = cursor.fetchone()[0]
        net = price - bp
        cursor.execute(f'UPDATE stocks SET net = "{net}" WHERE symbol="{stock}";')
    else: # entering a short position
        update_position(stock, "SHORT")
        cursor.execute(f'UPDATE stocks SET bought_price = "0.00" WHERE symbol="{stock}";')
        cursor.execute(f'UPDATE stocks SET net = "0.00" WHERE symbol="{stock}";')
    cursor.execute(f'UPDATE stocks SET sold_price = "{price}" WHERE symbol="{stock}";')
    connection.commit()
    connection.close()
    print(f'{currentTime()}: Updated stock {stock} prices successfully.')



def delete_stock(stock):
    connection, cursor = database()
    delete_command = f'DELETE FROM stocks WHERE symbol="{stock}";'
    cursor.execute(delete_command)
    connection.commit()
    connection.close()
    print(f'{currentTime()}: Deleted stock {stock} successfully.')

def create_report(email=False):
    """
    Create a report of all stock information in the db. This is either saved as a txt file locally (default) or
    is emailed to the provied email address.
    """
    report_time = datetime.datetime.strftime(datetime.datetime.utcnow(), "%m%d%Y_%H%M")
    # Retrieve and format the information
    stocks = get_all_stock_info()
    lines = []
    # Append header information
    lines.append(f'Daily Tradebot Report - Created at {report_time}\n')
    lines.append(f'Alpaca Account Cash: ${checkCash()}\n')
    lines.append(f'Alpaca Account Equity: ${checkEquity()}\n')
    lines.append(f'24-hour Gain/Loss: ${gainloss()}\n')
    lines.append("Individual Stock Details:\n")
    lines.append("--------------------------------------\n")
    for stock in stocks:
        line = "{0:{9}} {1:{10}} {2:{14}} {3:{11}} {4} {5:{12}} {6:{12}} {7:{13}} {8}\n".format(stock[0],stock[1],stock[2],stock[3],stock[4],
                                                                                                            stock[5],stock[6],stock[7],stock[8], 2, 5, 4, 5, 6, 5)
        lines.append(line)
    # Email, if requested.
    if email:
        message = ""
        for line in lines:
            message = message + line
        send_email(message)
    else:
        # write to the file, saved locally by default
        with open("stock_report_" + report_time + ".txt", "w") as report:
            report.writelines(lines)
        report.close()
