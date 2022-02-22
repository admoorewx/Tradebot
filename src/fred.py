from fredapi import Fred

fred = Fred(api_key='933b2e3bb59ae3431ef8d76e156444dc')

def getSeries(series):
	return fred.get_series(series)