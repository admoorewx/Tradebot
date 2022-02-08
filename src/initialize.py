import stocks_db as STD

# Enter the path to the stock_list.json file here
json_path = ""

STD.database()
STD.initialize()
STD.add_stock_from_json(json_path)
