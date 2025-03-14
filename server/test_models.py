import pickle
from datetime import date
import copy
import pytest
from models.forecast.attention_lstm import AttentionLSTM
from models.forecast.cnn_lstm import CNNLSTMTransformer
from models.lstm_attention import AttentionLstm
from models.forecast.transformer import ZavTransformer
from models.forecast.azad import AzSarima
from models.forecast.xgboost import XGBoost
from models.zav2 import Transformer


f = open('stock_data.pkl', 'rb')

data = pickle.load(f)

f.close()


# Testing AttentionLSTM
def test_attention_lstm():
    model = AttentionLSTM(AttentionLstm(), "attention_lstm", "TEST")
    model.train(copy.deepcopy(data))
    output = model.run(copy.deepcopy(data), 7)

    assert type(output[0]) is float

# Testing CNNLSTMTransformer
def cnnlstmtransformer():
    model = CNNLSTMTransformer("cnn-lstm", "TEST")
    model.train(copy.deepcopy(data))
    output = model.run(copy.deepcopy(data), 7)
    
    assert type(output[0]) is float

# Testing ZavTransformer
def test_transformer():
    model = ZavTransformer(Transformer(), "transformer", "TEST")
    model.train(copy.deepcopy(data))
    output = model.run(copy.deepcopy(data), 7)

    assert type(output[0]) is float

# Testing AzSarima
def test_azsarima():
    model = AzSarima("az-sarima", "TEST")
    model.train(copy.deepcopy(data))
    output = model.run(copy.deepcopy(data), 7)

    assert type(output[0]) is float

# Testing XGBoost, needs a real ticker name
def test_xgboost():
    model = XGBoost("XGBoost-model", "TSLA")
    model.train(copy.deepcopy(data))
    output = model.run(copy.deepcopy(data), 7)

    assert type(output[0]) is float
    


