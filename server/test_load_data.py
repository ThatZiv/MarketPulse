import pickle
import pytest
import load_data as ld
import database.reddit as reddit
import database.ddg_news as news
import database.yfinanceapi as yf

def mock_session():
    class session():
        def __init__(self, stuff =0):
            self.stuff=stuff
        def close(a):
            stuff = a
        def flush(a):
            stuff = a
        def add(a,b):
            stuff = a
        def commit(a):
            stuff = a
    return session()

def mock_valid_ticker(a, b):
    with open("test_data/load_data_stocks.pkl", "rb") as f:
        return pickle.load(f)
    return 0

def mock_valid_reddit(a, b):
    with open("test_data/load_data_reddit.pkl", "rb") as f:
        return pickle.load(f)
    return 0

def mock_valid_news(a):
    with open("test_data/load_data_news.pkl", "rb") as f:
        return pickle.load(f)
    return 0

def mock_empty_news(a):
    return []

def mock_empty_reddit(a, b):
    return []

def mock_empty_stocks(a, b):
    return []

def mock_yf(a, b):
    with open("test_data/load_data_yf.pkl", "rb") as f:
        return pickle.load(f)
    return 0

def test_all_inputs(monkeypatch):
    
    monkeypatch.setattr(ld, "create_session", mock_session)
    monkeypatch.setattr(ld, "stock_query_all", mock_valid_ticker)
    monkeypatch.setattr(reddit, "daily_reddit_request", mock_valid_reddit)
    monkeypatch.setattr(news, "add_news", mock_valid_news)
    monkeypatch.setattr(yf, "add_daily_data", mock_yf)

    ld.load_stocks()

def test_no_news(monkeypatch):
    monkeypatch.setattr(ld, "create_session", mock_session)
    monkeypatch.setattr(ld, "stock_query_all", mock_valid_ticker)
    monkeypatch.setattr(reddit, "daily_reddit_request", mock_valid_reddit)
    monkeypatch.setattr(news, "add_news", mock_empty_news)
    monkeypatch.setattr(yf, "add_daily_data", mock_yf)

    ld.load_stocks()

def test_no_reddit(monkeypatch):
    monkeypatch.setattr(ld, "create_session", mock_session)
    monkeypatch.setattr(ld, "stock_query_all", mock_valid_ticker)
    monkeypatch.setattr(reddit, "daily_reddit_request", mock_empty_reddit)
    monkeypatch.setattr(news, "add_news", mock_valid_news)
    monkeypatch.setattr(yf, "add_daily_data", mock_yf)

    ld.load_stocks()

def test_only_stocks(monkeypatch):
    monkeypatch.setattr(ld, "create_session", mock_session)
    monkeypatch.setattr(ld, "stock_query_all", mock_valid_ticker)
    monkeypatch.setattr(reddit, "daily_reddit_request", mock_empty_reddit)
    monkeypatch.setattr(news, "add_news", mock_empty_news)
    monkeypatch.setattr(yf, "add_daily_data", mock_yf)

    ld.load_stocks()

def test_no_stocks(monkeypatch):
    monkeypatch.setattr(ld, "create_session", mock_session)
    monkeypatch.setattr(ld, "stock_query_all", mock_empty_stocks)
    monkeypatch.setattr(reddit, "daily_reddit_request", mock_empty_reddit)
    monkeypatch.setattr(news, "add_news", mock_empty_news)
    monkeypatch.setattr(yf, "add_daily_data", mock_yf)

    ld.load_stocks() 
