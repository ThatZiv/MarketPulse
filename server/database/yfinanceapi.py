from pandas.core.indexes import period
import yfinance as yf

import pandas as pd




def bulk_stock_data(ticker):
    data = yf.Ticker(ticker)

    output = data.history(period = '2y')

    output.reset_index(inplace = True)

    return output

def add_daily_data(ticker, days):
    data = yf.Ticker(ticker)

    output = data.history(period=f"{days}d")

    output.reset_index(inplace=True)

    return output 


def real_time_data(ticker):
    data = yf.Ticker(ticker)
    output = data.history(period = "1d", interval = "1m")

    output.reset_index(inplace=True)

    return output


