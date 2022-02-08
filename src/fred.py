from fredapi import Fred

fred = Fred(api_key='')

def getSeries(series):
	return fred.get_series(series)
