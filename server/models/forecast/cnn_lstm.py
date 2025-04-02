
import copy
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, select, exc
from sqlalchemy.orm import sessionmaker
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, LSTM, BatchNormalization, Input, Multiply, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from models.forecast.forecast_types import DataForecastType, DatasetType
from models.forecast.model import ForecastModel
from database.tables import Stock_Info


class CNNLSTMTransformer(ForecastModel):
    """
    CNNLSTMTransformer implementation of ForecastModel abstract class
    """
    def __init__(self, name: str, ticker: str = None):
        self.seq_length = 6 
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model()
        super().__init__(self.model, name, ticker)

    def _build_model(self):
        inputs = Input(shape=(self.seq_length, 3)) 
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        lstm_out = LSTM(units=128, return_sequences=True)(x)
        lstm_out = self._attention(lstm_out)
        lstm_out = LSTM(units=64, return_sequences=False)(lstm_out)
        x = Dropout(0.5)(lstm_out)
        x = Dense(units=100, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(units=1)(x)
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
        return model

    def _attention(self, inputs):
        query = Dense(64, activation="tanh")(inputs)
        attention_scores = Dense(1)(query)
        context_vector = Multiply()([inputs, attention_scores])
        context_vector = Dense(inputs.shape[-1])(context_vector)
        return Add()([inputs, context_vector])

    def train(self, data_set: DatasetType):

        scaled_data = self.scaler.fit_transform(data_set[['Close', 'Sentiment_Data', 'News_Data']])
        X, y = self._create_sequences(scaled_data, self.seq_length)
        split_index = int(len(X) * 0.8)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                       callbacks=[early_stopping])

    def run(self, input_data: DatasetType, num_forecast_days: int) -> DataForecastType:

        scaled_data = self.scaler.transform(input_data[['Close', 'Sentiment_Data', 'News_Data']])
        X_test, _ = self._create_sequences(scaled_data, self.seq_length)

        predictions = []
        last_sequence = scaled_data[-self.seq_length:]
        sentiment = input_data['Sentiment_Data']
        news = input_data['News_Data']
        average_sentiment = sentiment.mean()
        average_news = news.mean()
        sentiment_adjustment = 0.04 * (sentiment.iloc[-1] - average_sentiment)
        news_adjustment = 0.02 * (news.iloc[-1] - average_news)

        for _ in range(num_forecast_days):
            prediction = self.model.predict(last_sequence.reshape(1, self.seq_length, 3))
            adjusted_prediction = prediction[0, 0] * (1 + sentiment_adjustment + news_adjustment)
            predictions.append(adjusted_prediction)
            last_sequence = np.append(last_sequence[1:], [[adjusted_prediction, sentiment.iloc[-1], news.iloc[-1]]], axis=0)

        predictions = np.array(predictions).reshape(-1, 1)
        predictions_actual = self.scaler.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), 2)))))[:, 0]

        return predictions_actual.tolist()

    def _create_sequences(self, data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)


if __name__ == "__main__":
    load_dotenv()
    USER = os.getenv("user")
    PASSWORD = os.getenv("password")
    HOST = os.getenv("host")
    PORT = os.getenv("port")
    DBNAME = os.getenv("dbname")

    DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"

    engine = create_engine(DATABASE_URL)
    try:
        with engine.connect() as connection:
            print("Connection successful!")
    except exc.OperationalError as e:
        print(e)
    except exc.TimeoutError as e:
        print(e)

    stock_q = select(Stock_Info).where(Stock_Info.stock_id == 1)
    Session = sessionmaker(bind=engine)
    session = Session()
    data2 = session.connection().execute(stock_q).all()

    s_open = []
    s_close = []
    s_high = []
    s_low = []
    s_volume = []
    s_sentiment = []
    s_news = []
    for row in data2:
        s_open.append(row[3])
        s_close.append(row[1])
        s_high.append(row[4])
        s_low.append(row[5])
        s_volume.append(row[2])
        s_sentiment.append(row[6]) 
        s_news.append(row[8])  

    data2 = {
        'Close': s_close,
        'Open': s_open,
        'High': s_high,
        'Low': s_low,
        'Volume': s_volume,
        'Sentiment_Data': s_sentiment,
        'News_Data': s_news
    }
    data2 = pd.DataFrame(data2)
    data_copy = copy.deepcopy(data2)

    model = CNNLSTMTransformer("cnn-lstm", "TSLA")
    model.train(data2)
    print(model.run(data_copy, 7))