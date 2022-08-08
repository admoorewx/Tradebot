# Tradebot

BTC-USD Prediction and Trading

Tradebot is now using a Recurrent Neural Network to predict hourly changes in price of only BTC-USD. This focus on a single asset is a proof-of-concept that
an RNN can make skillful (and profitable) predictions. Trading BTC avoids PTD rules, greatly simplifies the trading decision making, and negates the need for the SQL database. Additional assets may be introduced in future updates if this proof-of-concept works. 

This tool uses a combination of the Alpaca API (https://alpaca.markets/) for brokerage purposes, the yfinance API (https://github.com/ranaroussi/yfinance) to retrive stock data, and Tensorflow (https://www.tensorflow.org/) as the machine learning library.

Dependency requirements can be found in the requirements.txt file. 

This is still a work in progress; future changes are likely.
