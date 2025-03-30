# pylint: disable=all
import os as os
import dotenv
import pickle
import pytest
import flask_jwt_extended as jw
import database.yfinanceapi as yf
import routes.auth as a
import main as m
from main import app

def mock_base(a):
    return 1

def mock_session():
    class session():
        def __init__(self, stuff =0):
            self.stuff=stuff
        def close(a):
            stuff = 1
    return session()

def mock_ticker(query, session):
    class Output():
        def __init__(self, stock_id=1):
            self.stock_id = stock_id
    return Output()

def mock_enviorn(a):
    return "secret_key"

def mock_env():
    return "testing"

@pytest.fixture
def client(monkeypatch):
    with app.test_client() as client:
        with app.app_context():
            #Not the real jwt secret generated with node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
            app.config["JWT_SECRET_KEY"] = "8041d3df0459ab455e1c79394b6335911ef96927e821bab8479daf93efc60b64"
            yield client

def test_logo(client):
    response = client.get('/auth/logo')
    assert response.status_code == 401

    access_token = jw.create_access_token('testuser')
    headers = { 'Authorization': 'Bearer {}'.format(access_token) }
    response = client.get('/auth/logo', headers = headers)
    assert response.status_code == 400

    response = client.get('/auth/logo?ticker=TSLA', headers = headers)
    assert response.status_code == 200
    assert response.data != None

def test_stockrealtime(client, monkeypatch):
    monkeypatch.setattr(m, "create_session", mock_session)
    monkeypatch.setattr(m, "stock_query_single", mock_ticker)
    response = client.get('/stockrealtime')
    assert response.status_code == 401

    access_token = jw.create_access_token('testuser')
    headers = { 'Authorization': 'Bearer {}'.format(access_token) }
    response = client.get('/stockrealtime', headers = headers)
    assert response.status_code == 400

    with open("test_data/rt_stock_data.pkl", "rb") as f:
        monkeypatch.setattr(yf, "real_time_data", pickle.load(f))
        response = client.get('/stockrealtime?ticker=TSLA', headers = headers)
        assert response.status_code == 200
        assert response.json != None

def test_stockchart(client, monkeypatch):
    def mock_stock_chart(a, b):
        f = open('test_data/stock_chart', 'rb')
        data = pickle.load(f)
        f.close()
        return data
    monkeypatch.setattr(a, "create_session", mock_session)
    monkeypatch.setattr(a, "stock_query_single", mock_ticker)
    monkeypatch.setattr(a, "stock_query_all", mock_stock_chart)

    response = client.get('/auth/stockchart')
    assert response.status_code == 401

    access_token = jw.create_access_token('testuser')
    headers = { 'Authorization': 'Bearer {}'.format(access_token) }
    response = client.get('/auth/stockchart', headers = headers)
    assert response.status_code == 400
    assert response.json == None

    response = client.get('/auth/stockchart?ticker=TSLA', headers = headers)
    assert response.status_code == 200
    assert response.json != None

def test_forecast(client, monkeypatch):
    def mock_forecast(a, b):
        f = open('test_data/forecast', 'rb')
        data = pickle.load(f)
        f.close()
        return data

    def mock_forecast_all(a, b):
        f = open('test_data/forecast2', 'rb')
        data = pickle.load(f)
        f.close()
        return data

    monkeypatch.setattr(a, "create_session", mock_session)
    monkeypatch.setattr(a, "stock_query_single", mock_forecast)
    monkeypatch.setattr(a, "stock_query_all", mock_forecast_all)
    response = client.get('/auth/forecast')
    assert response.status_code == 401

    access_token = jw.create_access_token('testuser')
    headers = { 'Authorization': 'Bearer {}'.format(access_token) }
    response = client.get('/auth/forecast', headers = headers)
    assert response.status_code == 400

    response = client.get('/auth/forecast?ticker=F&lookback=7', headers = headers)
    assert response.status_code == 200
    assert response.json != None
