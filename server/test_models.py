# pylint: disable=all
import pickle
import copy
from models.forecast.attention_lstm import AttentionLSTM
from models.forecast.cnn_lstm import CNNLSTMTransformer
from models.lstm_attention import AttentionLstm
from models.forecast.transformer import ZavTransformer
from models.forecast.azad import AzSarima
from models.forecast.xgboost import XGBoost
from models.zav2 import Transformer


f = open('test_data/stock_data.pkl', 'rb')
data = pickle.load(f)
f.close()


# Testing AttentionLSTM
def test_attention_lstm():
    model = AttentionLSTM(AttentionLstm(), "attention_lstm", "TEST")
    model.train(copy.deepcopy(data))
    output = model.run(copy.deepcopy(data), 7)

    assert all(isinstance(val, float) for val in output)

# Testing CNNLSTMTransformer
def cnnlstmtransformer():
    model = CNNLSTMTransformer("cnn-lstm", "TEST")
    model.train(copy.deepcopy(data))
    output = model.run(copy.deepcopy(data), 7)
    assert isinstance(output[0], float)

# Testing ZavTransformer
def test_transformer():
    model = ZavTransformer(Transformer(), "transformer", "TEST")
    model.train(copy.deepcopy(data))
    output = model.run(copy.deepcopy(data), 7)

    assert isinstance(output[0],float)

# Testing AzSarima
def test_azsarima():
    model = AzSarima("az-sarima", "TEST")
    model.train(copy.deepcopy(data))
    output = model.run(copy.deepcopy(data), 7)

    assert isinstance(output[0], float)

# Testing XGBoost, needs a real ticker name
def test_xgboost():
    model = XGBoost("XGBoost-model", "TSLA")
    model.train(copy.deepcopy(data))
    output = model.run(copy.deepcopy(data), 7)

    assert isinstance(output[0], float)
