# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import yfinance as yf




def bulk_stock_data(ticker):
    '''Function used to get the past 2 years'''
    data = yf.Ticker(ticker)

    output = data.history(period = '2y')

    output.reset_index(inplace = True)

    return output

def add_daily_data(ticker, days):
    '''get the past #days of stock data.
    Returns an array of stock information that can be stored in the database'''
    data = yf.Ticker(ticker)

    output = data.history(period=f"{days}d")

    output.reset_index(inplace=True)

    return output

def real_time_data(ticker):
    '''Used to get real tune stock data for a ticker with 1m interval'''
    data = yf.Ticker(ticker)
    output = data.history(period = "1d", interval = "1m")

    output.reset_index(inplace=True)

    return output
