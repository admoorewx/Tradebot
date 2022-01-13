import sqlite3
import json
from functions import currentTime

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
    owned: interger, 1 = True (owned), 0 = False (not owned).
    last_transaction: string, either "buy", "sell" or "passed".
    last_transaction_time: string, datetime string of last buy/sell transaction.
    transaction_price: real (float), the stock price at the time of the last transaction.
    transaction_reason: string, reason why the last transaction was executed.
    """
    create_talbes = """CREATE TABLE stocks (
    stock_number INTEGER PRIMARY KEY,
    symbol TEXT,
    owned INTEGER,
    last_transaction TEXT,
    last_transaction_time TEXT,
    transaction_price REAL,
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
            add_command = f'INSERT INTO stocks VALUES("{i}","{stock}","0","Pass","01/01/2022 00:00","0.00","initial");'
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
    cursor.execute(f'SELECT symbol, owned, last_transaction, last_transaction_time, transaction_price, transaction_reason FROM stocks WHERE symbol = "{stock}";')
    response = cursor.fetchone()
    connection.close()
    return response

def update_stock(stock,owned,transaction,transaction_time,transaction_price,reason):
    connection, cursor = database()
    update_owned = f'UPDATE stocks SET owned = "{owned}" WHERE symbol="{stock}";'
    update_transaction = f'UPDATE stocks SET last_transaction = "{transaction}" WHERE symbol="{stock}";'
    update_transaction_time = f'UPDATE stocks SET last_transaction_time = "{transaction_time}" WHERE symbol="{stock}";'
    update_transaction_price = f'UPDATE stocks SET transaction_price = "{transaction_price}" WHERE symbol="{stock}";'
    update_reason = f'UPDATE stocks SET transaction_reason = "{reason}" WHERE symbol="{stock}";'
    commands = [update_owned, update_transaction, update_transaction_time, update_transaction_price, update_reason]
    for command in commands:
        cursor.execute(command)
    connection.commit()
    connection.close()
    print(f'{currentTime()}: Updated stock {stock} successfully.')

def delete_stock(stock):
    connection, cursor = database()
    delete_command = f'DELETE FROM emp WHERE symbol="{stock}";'
    cursor.execute(delete_command)
    connection.commit()
    connection.close()
    print(f'{currentTime()}: Deleted stock {stock} successfully.')



