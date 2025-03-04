import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit,cross_val_score
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import plot_importance
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score 
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

class XGBoostModel:
    def __init__(self,ticker):
        self.ticker = ticker

    def add_features(self, df):
        if self.ticker == 'TSLA': 
            df = self.tsla_features(df)
            print('Added TSLA features')
        elif self.ticker == 'GM':
            df = self.gm_features(df)
            print('Added GM features')
        elif self.ticker == 'F':
            df = self.f_features(df)
            print('Added F features')
        elif self.ticker == 'TM':
            df = self.tm_features(df)
            print('Added TM features')
        elif self.ticker == 'STLA':
            df = self.stla_features(df)
            print('Added STLA features')            
        return df

    def tsla_features(self, frame):
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

        # frame["diff_1"] = frame["Close"].diff(1)  
        # frame["diff_7"] = frame["Close"].diff(7)
        return frame 
    
    def gm_features(frame):
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

    def ford_features(frame):
        frame = frame.copy()
        if 'Close' not in frame.columns:
            print("Error: 'Close' column not found in the DataFrame!")
            return None
        frame['Day of Year'] = frame.index.dayofyear
        frame['Week Of Year'] = frame.index.isocalendar().week
        frame['Last Day Price'] = frame['Close'].shift(1)
        frame['Last Week Price'] = frame['Close'].shift(7)
        frame['3 days ago'] = frame['Close'].shift(3)
        frame['2 weeks ago'] = frame['Close'].shift(14)
        frame['rolling_mean_7'] = frame['Close'].shift(1).rolling(window=7).mean()
        frame['rolling_mean_30'] = frame['Close'].shift(1).rolling(window=30).mean()
        frame["diff_1"] = frame["Close"].shift(1).diff()  
        frame["diff_7"] = frame["Close"].shift(7).diff(7) 
        return frame
    
    def tm_features(frame):
        frame = frame.copy()
        if 'Close' not in frame.columns:
            print("Error: 'Close' column not found in the DataFrame!")
            return None
        frame['Week Of Year'] = frame.index.isocalendar().week
        frame['Last Day Price'] = frame['Close'].shift(1)
        frame['Last Week Price'] = frame['Close'].shift(7)
        frame['3 days ago'] = frame['Close'].shift(3)
        frame['2 weeks ago'] = frame['Close'].shift(14)
        frame['rolling_mean_30'] = frame['Close'].shift(1).rolling(window=30).mean()
        return frame

    def stla_features(frame):
        frame['Last Day Price'] = frame['Close'].shift(1)
        frame['Last Week Price'] = frame['Close'].shift(7)
        frame['3 days ago'] = frame['Close'].shift(3)
        frame['2 weeks ago'] = frame['Close'].shift(14)
        frame['rolling_mean_7'] = frame['Close'].shift(1).rolling(window=7).mean()
        frame['rolling_std_7'] = frame['Close'].shift(1).rolling(window=7).std()
        return frame
    
    def test_train_split(self, df):
        df = df.copy()
        df.fillna(df.mean(), inplace=True)
        X = df.drop(columns=['Close'])
        y = df['Close']
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy().ravel()
        else:
            y = y.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        print("XGBoost Train/Test split completed.")
        return X_train, X_test, y_train, y_test
    
    # used for both testing and predictions.
    def model_train(self, X, y):
        print("XGBoost Training Started.")
        xgb = XGBRegressor(objective='reg:squarederror', booster='gbtree',reg_alpha=1,reg_lambda=9, gamma=5)
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
    
    def model_predictions(self, model, X):
        print("XGBoost Predictions:")
        return model.predict(X)
    
    def model_test_run(self, df):
        df = df.copy()
        X_train, X_test, y_train, y_test = self.test_train_split(df)
        model = self.model_train(X_train,y_train)
        predictions = self.model_predictions(model, X_test)
        print("XGBoost Test Run Completed.")
        return predictions, y_test
    
    def model_actual_run(self, df):
        df = df.copy()
        drop_cols = ['Open','High','Low','Volume']
        df = df.drop(columns=drop_cols)
        df = self.add_features(df)
        df.fillna(df.mean(), inplace=True)
        X = df.drop(columns='Close')
        y = df['Close']
        model = self.model_train(X, y)
        return model

    def future_predictions(self, model, df, num_days):
        print("Continuing with Future Predictions:")
        df = df.copy()
        today = date.today()
        future = pd.date_range(today, periods=1)
        future_df = pd.DataFrame(index=future)
        future_df['isFuture'] = True
        df['isFuture'] = False
        first_drop_cols = ['Low','High','Open','Volume']
        df = df.drop(columns=first_drop_cols)
        df_and_future = pd.concat([df, future_df])
        df_and_future = df_and_future.fillna(value=0)
        df_and_future = self.add_features(df_and_future)
        future_values = df_and_future.query('isFuture').copy()
        drop_cols = ['Close','isFuture']
        future_values = future_values.drop(columns=drop_cols)
        holidays_dates = pd.to_datetime(['2025-01-01','2025-01-09'])
        print(f"00000: {df_and_future}")

        for i in range(num_days):
            next_day_pred = self.model_predictions(model, future_values)
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
            print(f"{i}: {df_and_future}")
        predicted = df_and_future.tail(num_days+1).iloc[:-1]['Close']
        print(f"Predicted prices for requested days: {num_days}")
        return predicted

    def model_evaluation(self, predictions, y_test):
        print("XGBoost Model Evaluation:")
        predictions = np.ravel(predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        metrics = {
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse,
            'Mean Absolute Percentage Error': mape,
            'R^2 Score': r2
        }
        plt.figure(figsize=(10,5))
        plt.plot(y_test, label='Actual Price', color='blue', linestyle='-')
        plt.plot(predictions, label='Predicted Price', color='red', linestyle='-')
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.title('Actual vs Forecast')
        plt.show()
        return metrics
    
if __name__ == "__main__":
    today = date.today()
    data = yf.download('TSLA', start='2020-01-01', end=today.strftime('%Y-%m-%d'))
    data.columns = data.columns.get_level_values(0)
    model = XGBoostModel('TSLA')
    print(data)
    # Train/Test Evaluation
    # predictions, true = model.model_test_run(data)
    # for i in range(len(predictions)):
    #     print(predictions[i], true[i], end='\n')
    # eval = model.model_evaluation(predictions, true)
    # print(eval)

    # Future Predictions
    optimal_model = model.model_actual_run(data)
    predictions = model.future_predictions(optimal_model, data, 7)
    print("Predictions:\n")
    for i in range(len(predictions)):
        print(predictions[i], end='\n')