# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import yfinance as yf




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
