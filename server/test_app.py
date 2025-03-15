import sys
import os
import pytest
from main import app
import flask_jwt_extended as jw
from unittest.mock import patch
import pickle
import database.yfinanceapi as yf
from unittest import mock
import sqlalchemy.orm as db
from sqlalchemy import select

import engine as eg




@pytest.fixture
def mock_session():
    mock_session = mock.Mock()

    mock_connection = mock.Mock()
    mock_session.connection.return_value = mock_connection

    mock_result = mock.Mock()
    mock_result.first.return_value = ('mocked_id',)

    mock_connection.execute.return_value = mock_result

    return mock_session
@pytest.fixture
def client():
    with app.test_client() as client:
        with app.app_context():
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

def test_stockrealtime(client, monkeypatch, mock_session):
    
    monkeypatch.setattr(eg, "get_engine", True)
    monkeypatch.setattr(db ,'sessionmaker', mock.Mock(return_value=mock_session))

    response = client.get('/stockrealtime')
    assert response.status_code == 401

    access_token = jw.create_access_token('testuser')
    headers = { 'Authorization': 'Bearer {}'.format(access_token) }
    response = client.get('/stockrealtime', headers = headers)
    assert response.status_code == 400

    with open("rt_stock_data.pkl", "rb") as f:
        monkeypatch.setattr(yf, "real_time_data", pickle.load(f))
        response = client.get('/stockrealtime?ticker=TSLA', headers = headers)
        assert response.status_code == 200
        assert response.json != None


def test_stockchart(client):
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


def test_forecast(client):
    response = client.get('/auth/forecast')
    assert response.status_code == 401

    access_token = jw.create_access_token('testuser')
    headers = { 'Authorization': 'Bearer {}'.format(access_token) }
    response = client.get('/auth/forecast', headers = headers)
    assert response.status_code == 400

    response = client.get('/auth/forecast?ticker=TSLA', headers = headers)
    assert response.status_code == 200
    assert response.json != None

