# pylint: disable=all
import os
import pickle
from unittest.mock import patch
import requests
import dotenv
import torch
from pandas import Timestamp
import database.reddit as reddit



def mock_env():
    return "testing"

def mock_enviorn(a):
    return "secret_key"

def mock_auth():
    return "Test"

def test_base_request(monkeypatch):
    output = 5
    def mock_get(url, headers, params, timeout):
        return output

    monkeypatch.setattr(dotenv, "load_dotenv", mock_env)
    monkeypatch.setattr(os.environ, "get", mock_enviorn)
    monkeypatch.setattr(reddit, "reddit_auth", mock_auth)
    monkeypatch.setattr(requests , "get", mock_get)

    assert reddit.base_request("url", "params", "Auth") == 5
    class MockTimeoutError:
        def __init__(self, *args, **kargs):
            raise requests.exceptions.Timeout

    monkeypatch.setattr(requests , "get", MockTimeoutError)
    assert reddit.base_request("url", "params", "Auth")==-1

    class MockHttpError:
        def __init__(self, *args, **kargs):
            raise requests.exceptions.Timeout

    monkeypatch.setattr(requests , "get", MockHttpError)
    assert reddit.base_request("url", "params", "Auth")==-1

    class MockConnectionError:
        def __init__(self, *args, **kargs):
            raise requests.ConnectionError

    monkeypatch.setattr(requests , "get", MockConnectionError)
    assert reddit.base_request("url", "params", "Auth")==-1

    class MockRequestException:
        def __init__(self, *args, **kargs):
            raise requests.exceptions.RequestException

    monkeypatch.setattr(requests , "get", MockRequestException)
    assert reddit.base_request("url", "params", "Auth")==-1

def test_request_comment(monkeypatch):
    with open("test_data/request_comment.pkl", "rb") as f:
        output = pickle.load(f)

    def mock_get(url, headers, params, timeout):
        return output

    monkeypatch.setattr(dotenv, "load_dotenv", mock_env)
    monkeypatch.setattr(os.environ, "get", mock_enviorn)
    monkeypatch.setattr(reddit, "reddit_auth", mock_auth)

    monkeypatch.setattr(requests , "get", mock_get)

    # outputs the json when given a correct response
    assert reddit.request_comment("url", "name", 10, 'Auth') != -1

    output = 5
    assert reddit.request_comment("url", "name", 10, 'Auth') == -1

def test_reddit_request(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", mock_env)
    monkeypatch.setattr(os.environ, "get", mock_enviorn)
    monkeypatch.setattr(reddit, "reddit_auth", mock_auth)

    with open("test_data/request_search.pkl", "rb") as f1, open("test_data/request_after.pkl", "rb") as f2, open("test_data/request_comment_json.pkl", "rb") as f3:
        with patch('database.reddit.request_search', return_value = pickle.load(f1)) as mock_1, patch('database.reddit.request_after', return_value = pickle.load(f2)) as mock_2, patch('database.reddit.request_comment', return_value = pickle.load(f3)) as mock_3:
            # output has 200 status code
            assert reddit.reddit_request("subreddit", "topic", 1, 1, "datec") != 0


def test_daily_reddit_request(monkeypatch):
    monkeypatch.setattr(reddit, "reddit_auth", mock_auth)
    dates = [{'time_stamp': Timestamp('2025-03-15 00:00:00-0400', tz='America/New_York'), 'ticker': 'TSLA', 'search': 'Tesla', 'stock_id': 1, 'stock_close': 240.67999267578125, 'stock_volume': 111894747, 'stock_open': 248.125, 'stock_high': 248.2899932861328, 'stock_low': 233.52999877929688, 'news_data': 0, 'sentiment_data': 0}]

    with open("test_data/request_search.pkl", "rb") as f1, open("test_data/request_after.pkl", "rb") as f2, open("test_data/request_comment_json.pkl", "rb") as f3:
        with patch('models.sentiment_hf.sentiment_model', return_value = torch.tensor([[1,3,.5]])) as mock_4, patch('database.reddit.request_search', return_value = pickle.load(f1)) as mock_1, patch('database.reddit.request_after', return_value = pickle.load(f2)) as mock_2, patch('database.reddit.request_comment', return_value = pickle.load(f3)) as mock_3:
            dates = [{'time_stamp': Timestamp('2025-03-15 00:00:00-0400', tz='America/New_York'), 'ticker': 'TSLA', 'search': 'Tesla', 'stock_id': 1, 'stock_close': 240.67999267578125, 'stock_volume': 111894747, 'stock_open': 248.125, 'stock_high': 248.2899932861328, 'stock_low': 233.52999877929688, 'news_data': 0, 'sentiment_data': 0}]
            assert not reddit.daily_reddit_request("test", dates)
            dates = [{'time_stamp': Timestamp('2025-03-13 00:00:00-0400', tz='America/New_York'), 'ticker': 'TSLA', 'search': 'Tesla', 'stock_id': 1, 'stock_close': 240.67999267578125, 'stock_volume': 111894747, 'stock_open': 248.125, 'stock_high': 248.2899932861328, 'stock_low': 233.52999877929688, 'news_data': 0, 'sentiment_data': 0}]
            output =  reddit.daily_reddit_request("test", dates)
            assert output[0]['time_stamp'] == Timestamp('2025-03-13 00:00:00-0400', tz='America/New_York')
