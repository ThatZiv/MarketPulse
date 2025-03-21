from datetime import date, timedelta
import pandas as pd
# import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class XGBoostModel:
    """
    This class implements an XGBoost model for training and prediction.
    """
    def __init__(self,ticker):
        self.ticker = ticker

    def add_features(self, df):
        """
        This function adds features to the DataFrame based on the ticker.
        """
        if self.ticker == 'TSLA':
            df = self.tsla_features(df)
        elif self.ticker == 'GM':
            df = self.gm_features(df)
        elif self.ticker == 'F':
            df = self.ford_features(df)
        elif self.ticker == 'TM':
            df = self.tm_features(df)
        elif self.ticker == 'STLA':
            df = self.stla_features(df)
        return df

    def tsla_features(self, frame):
        """
        This function adds TSLA features to the dataframe.
        """
        frame = frame.copy()
        if 'Close' not in frame.columns:
            print("Error: 'Close' column not found in the DataFrame!")
            return None
        frame['Last Day Price'] = frame['Close'].shift(1)
        frame['Last Week Price'] = frame['Close'].shift(7)
        frame['3 days ago'] = frame['Close'].shift(3)
        frame['2 weeks ago'] = frame['Close'].shift(14)
        frame['rolling_mean_7'] = frame['Close'].shift(1).rolling(window=7).mean()
        frame['rolling_mean_30'] = frame['Close'].shift(1).rolling(window=30).mean()
        frame['rolling_std_7'] = frame['Close'].shift(1).rolling(window=7).std()
        return frame

    def gm_features(self, frame):
        """
        This function adds GM features to the dataframe.
        """
        frame = frame.copy()
        if 'Close' not in frame.columns:
            print("Error: 'Close' column not found in the DataFrame!")
            return None
        frame['Last Day Price'] = frame['Close'].shift(1)
        frame['Last Week Price'] = frame['Close'].shift(7)
        frame['3 days ago'] = frame['Close'].shift(3)
        frame['2 weeks ago'] = frame['Close'].shift(14)
        frame['rolling_mean_7'] = frame['Close'].shift(1).rolling(window=7).mean()
        frame['rolling_mean_30'] = frame['Close'].shift(1).rolling(window=30).mean()
        return frame

    def ford_features(self, frame):
        """
        This function adds Ford features to the dataframe.
        """
        frame = frame.copy()
        if 'Close' not in frame.columns:
            print("Error: 'Close' column not found in the DataFrame!")
            return None
        frame['Last Day Price'] = frame['Close'].shift(1)
        frame['Last Week Price'] = frame['Close'].shift(7)
        frame['3 days ago'] = frame['Close'].shift(3)
        frame['2 weeks ago'] = frame['Close'].shift(14)
        frame['rolling_mean_7'] = frame['Close'].shift(1).rolling(window=7).mean()
        frame['rolling_mean_30'] = frame['Close'].shift(1).rolling(window=30).mean()
        frame["diff_1"] = frame["Close"].shift(1).diff()
        frame["diff_7"] = frame["Close"].shift(7).diff(7)
        return frame

    def tm_features(self, frame):
        """
        This function adds TM features to the dataframe.
        """
        frame = frame.copy()
        if 'Close' not in frame.columns:
            print("Error: 'Close' column not found in the DataFrame!")
            return None
        frame['Last Day Price'] = frame['Close'].shift(1)
        frame['Last Week Price'] = frame['Close'].shift(7)
        frame['3 days ago'] = frame['Close'].shift(3)
        frame['2 weeks ago'] = frame['Close'].shift(14)
        frame['rolling_mean_30'] = frame['Close'].shift(1).rolling(window=30).mean()
        return frame

    def stla_features(self, frame):
        """
        This function adds STLA features to the dataframe.
        """
        frame['Last Day Price'] = frame['Close'].shift(1)
        frame['Last Week Price'] = frame['Close'].shift(7)
        frame['3 days ago'] = frame['Close'].shift(3)
        frame['2 weeks ago'] = frame['Close'].shift(14)
        frame['rolling_mean_7'] = frame['Close'].shift(1).rolling(window=7).mean()
        frame['rolling_std_7'] = frame['Close'].shift(1).rolling(window=7).std()
        return frame

    def test_train_split(self, df):
        """
        This function splits the DataFrame into training and testing sets.
        Used mostly for training and testing the model.
        """
        df = df.copy()
        df.fillna(df.mean(), inplace=True)
        X = df.drop(columns=['Close']) # pylint: disable=invalid-name
        y = df['Close']
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy().ravel()
        else:
            y = y.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # pylint: disable=invalid-name
        print("XGBoost Train/Test split completed.")
        return X_train, X_test, y_train, y_test

    # used for both testing and predictions.
    def model_train(self, X, y): # pylint: disable=invalid-name
        """
        This function trains the model using the features and target dataframes.
        """
        print("XGBoost Training Started.")
        xgb = XGBRegressor(objective='reg:squarederror',
                           booster='gbtree',reg_alpha=1,reg_lambda=9, gamma=5)
        rf = RandomForestRegressor(random_state=12,n_jobs=-1 )
        lin_reg = LinearRegression()
        voting_model = VotingRegressor(estimators=[('xgb', xgb), ('rf',rf),('lin', lin_reg)])
        search_space = {
            'xgb__max_depth': Integer(2, 5),
            'xgb__learning_rate': Real(0.01, 1, prior='log-uniform'),
            'xgb__subsample': Real(0.1, 1.0),
            'xgb__colsample_bytree': Real(0.1, 1.0),
            'xgb__colsample_bylevel': Real(0.1, 1.0),
            'xgb__colsample_bynode': Real(0.1, 1.0),
            'xgb__n_estimators': Integer(100, 1000),
            'rf__n_estimators': Integer(50, 200),
            'rf__max_depth': Integer(5, 20),
            'rf__min_samples_split': Integer(2, 20), 
            'rf__min_samples_leaf': Integer(1, 20),  
            'rf__max_features': Real(0.1, 1.0), 
            'rf__max_leaf_nodes': Integer(10, 100), 
            'rf__min_impurity_decrease': Real(0.0, 0.1),
        }
        tscv = TimeSeriesSplit(n_splits=2)

        bayes_search = BayesSearchCV(
            estimator=voting_model,
            search_spaces=search_space,
            n_iter=50,
            cv=tscv,
            scoring='r2',
            n_jobs=1,
            random_state=12
        )
        bayes_search.fit(X, y)
        print("XGBoost Training Completed.")
        return bayes_search.best_estimator_

    def model_predictions(self, tr_model, X): # pylint: disable=invalid-name
        """
        This function makes predictions using the trained model and features.
        """
        return tr_model.predict(X)

    def model_test_run(self, df):
        """
        This function trains the model for testing and evaluation.
        """
        df = df.copy()
        X_train, X_test, y_train, y_test = self.test_train_split(df) # pylint: disable=invalid-name
        tr_model = self.model_train(X_train,y_train) # pylint: disable=invalid-name
        model_predictions = self.model_predictions(tr_model, X_test) # pylint: disable=invalid-name
        print("XGBoost Test Run Completed.")
        return model_predictions, y_test

    def model_actual_run(self, df):
        """
        This function trains the model for actual predictions.
        """
        df = df.copy()
        drop_cols = ['Open','High','Low','Volume','Sentiment_Data','News_Data']
        df = df.drop(columns=drop_cols)
        df = self.add_features(df)
        print(f"Added {self.ticker} features.")
        df.fillna(df.mean(), inplace=True)
        X = df.drop(columns='Close') # pylint: disable=invalid-name
        y = df['Close']
        tr_model = self.model_train(X, y)
        return tr_model

    def future_predictions(self, tr_model, df, num_days):
        """
        This function makes future predictions using the trained model.
        """
        print("Continuing with Future Predictions:")
        df = df.copy()
        date_now = date.today()
        if date_now.weekday() == 5:
            date_now += timedelta(days=2)
        elif date_now.weekday() == 6:
            date_now += timedelta(days=1)
        future = pd.date_range(date_now, periods=1)
        future_df = pd.DataFrame(index=future)
        future_df['isFuture'] = True
        df['isFuture'] = False
        first_drop_cols = ['Low','High','Open','Volume', 'Sentiment_Data', 'News_Data']
        df = df.drop(columns=first_drop_cols)
        df_and_future = pd.concat([df, future_df])
        df_and_future = df_and_future.fillna(value=0)
        df_and_future = self.add_features(df_and_future)
        future_values = df_and_future.query('isFuture').copy()
        drop_cols = ['Close','isFuture']
        future_values = future_values.drop(columns=drop_cols)
        holidays_dates = pd.to_datetime(['2025-01-01','2025-01-09'])

        for __ in range(num_days):
            next_day_pred = self.model_predictions(tr_model, future_values)
            df_and_future.at[df_and_future.index[-1], 'Close'] = next_day_pred
            df_and_future.at[df_and_future.index[-1], 'isFuture'] = False
            current_date = future[-1]
            custom_business_day = pd.offsets.CustomBusinessDay(holidays=holidays_dates)
            next_business_day = current_date + custom_business_day
            future = pd.date_range(next_business_day, next_business_day)
            future_df = pd.DataFrame(index=future)
            future_df['isFuture'] = True
            df_and_future = pd.concat([df_and_future, future_df])
            df_and_future = df_and_future.fillna(value=0)
            df_and_future = self.add_features(df_and_future)
            future_values = df_and_future.query('isFuture').copy()
            drop_cols = ['Close','isFuture']
            future_values = future_values.drop(columns=drop_cols)
        predicted = df_and_future.tail(num_days+1).iloc[:-1]['Close']
        return predicted

    def model_evaluation(self, y_pred, y_test):
        """
        This function evaluates the model's performance using the predictions and true values.
        """
        print("XGBoost Model Evaluation:")
        y_pred = np.ravel(y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        metrics = {
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse,
            'Mean Absolute Percentage Error': mape,
            'R^2 Score': r2
        }
        plt.figure(figsize=(10,5))
        plt.plot(y_test, label='Actual Price', color='blue', linestyle='-')
        plt.plot(y_pred, label='Predicted Price', color='red', linestyle='-')
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.title('Actual vs Forecast')
        plt.show()
        return metrics

if __name__ == "__main__":
    today = date.today()
    data = yf.download('TM', start='2020-01-01', end=today.strftime('%Y-%m-%d'))
    data.columns = data.columns.get_level_values(0)
    model = XGBoostModel('TM')
    print(data)
    # Train/Test Evaluation
    # predictions, true = model.model_test_run(data)
    # for i in range(len(predictions)):
    #     print(predictions[i], true[i], end='\n')
    # eval = model.model_evaluation(predictions, true)
    # print(eval)

    # Future Predictions
    # optimal_model = model.model_actual_run(data)
    # predictions = model.future_predictions(optimal_model, data, 7)
    # print("Predictions:\n")
    # for i in range(len(predictions)):
    #     print(predictions[i], end='\n')