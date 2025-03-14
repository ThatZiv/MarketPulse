import database.ddg_news as ddg
import pytest
import requests
import datetime
import dotenv
import os
import pickle
from pandas import Timestamp
from unittest.mock import patch
from duckduckgo_search import DDGS


def test_add_news():
    dates = [{'time_stamp': Timestamp('2025-03-13 00:00:00-0400', tz='America/New_York'), 'ticker': 'TSLA', 'search': 'Tesla', 'stock_id': 1, 'stock_close': 240.67999267578125, 'stock_volume': 111894747, 'stock_open': 248.125, 'stock_high': 248.2899932861328, 'stock_low': 233.52999877929688, 'news_data': 0, 'sentiment_data': 0}]
    with open("ddg_news.pkl", "rb") as f:
        with patch.object(DDGS, "text", return_value=pickle.load(f)) as mock_text:
            result = ddg.add_news(dates)
            assert result [0]['time_stamp'] == '2025-03-13'

