import sys
import os
import pytest
from main import app
import flask_jwt_extended as jw
from unittest.mock import patch


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
    
def test_stockrealtime(client):
    response = client.get('/stockrealtime')
    assert response.status_code == 401

    access_token = jw.create_access_token('testuser')
    headers = { 'Authorization': 'Bearer {}'.format(access_token) }
    response = client.get('/stockrealtime', headers = headers)
    assert response.status_code == 400


    response = client.get('/stockrealtime?ticker=TSLA', headers = headers)
    assert response.status_code == 200
    #assert response.json == None

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
    #assert response.json == None


def test_forecast(client):
    response = client.get('/auth/forecast')
    assert response.status_code == 401

    access_token = jw.create_access_token('testuser')
    headers = { 'Authorization': 'Bearer {}'.format(access_token) }
    response = client.get('/auth/forecast', headers = headers)
    assert response.status_code == 400

    response = client.get('/auth/forecast?ticker=TSLA', headers = headers)
    assert response.status_code == 200
    #assert response.json == None

