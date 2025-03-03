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

class XGBoostModel:
    def __init__(self, df, ticker):
        if ticker == 'TSLA': 
            self.df = self.tsla_features(df)

    def tsla_features(self, frame):
        frame = frame.copy()
        
        frame['Last Day Price'] = frame['Close'].shift(1)
        frame['Last Week Price'] = frame['Close'].shift(7)
        frame['3 days ago'] = frame['Close'].shift(3)
        frame['2 weeks ago'] = frame['Close'].shift(14)
        
        frame['rolling_mean_7'] = frame['Close'].shift(1).rolling(window=7).mean()
        frame['rolling_mean_30'] = frame['Close'].shift(1).rolling(window=30).mean()
        frame['rolling_std_7'] = frame['Close'].shift(1).rolling(window=7).std()

        frame["diff_1"] = frame["Close"].diff(1)  
        frame["diff_7"] = frame["Close"].diff(7)

        return frame 
    
    def test_train_split(self, df):
        X = df.drop(columns=['Close'])
        y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test
    
    # used for both testing and predictions.
    def model_train(self, X, y):
        xgb = XGBRegressor(objective='reg:squarederror', booster='gbtree',reg_alpha=1,reg_lambda=9, gamma=5)
        rf = RandomForestRegressor(random_state=12,n_jobs=-1 )
        lin_reg = LinearRegression()
        voting_model = VotingRegressor(estimators=[('xgb', xgb), ('rf',rf),('lin', lin_reg)])
        search_space = {
            'xgb__max_depth': Integer(2, 5),
            'xgb__learning_rate': Real(0.01, 1, prior='log-uniform'),  # Increased learning rate range
            'xgb__subsample': Real(0.1, 1.0),  # Slightly less aggressive subsampling
            'xgb__colsample_bytree': Real(0.1, 1.0), # Added possibility of column subsampling
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
        return bayes_search.best_estimator_
    
    def model_predictions(self, model, X):
        return model.predict(X)
    def model_test_run(self, df):
        X_train, X_test, y_train, y_test = self.test_train_split(df)
        model = self.model_train(X_train,y_train)
        predictions = self.model_predictions(model, X_test)
        return predictions, y_test

    
    def model_actual_run(self, model, df):
        df = df.copy()
        df = self.tsla_features(df)
        X = df.drop(columns='Close')
        y = df['Close']
        model = self.model_train(X, y)
        predictions = self.model_predictions(model, X)
        return predictions
    
    def model_evaluation(self, predictions, y_test):
        predictions = np.ravel(predictions)
        y_test = (y_test).to_numpy()
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        return mse, rmse, mape, r2
    
if __name__ == "__main__":
    data = yf.download('TSLA', start='2020-01-01', end='2024-12-31')
    model = XGBoostModel(data, 'TSLA')
    predictions, true = model.model_test_run(data)
    print(predictions, true, end='\n')
    eval = model.model_evaluation(predictions, true)
    print(eval)
    print(model[0] + " "+ model[1])