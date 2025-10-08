from numpy import sqrt
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def print_error(Y_test, Y_pred, custom = False):
    MAE = round(mean_absolute_error(Y_test, Y_pred) if custom else np.mean(np.abs(Y_test-Y_pred)), 5)
    MSE = round(mean_squared_error(Y_test, Y_pred) if custom else np.mean((Y_test-Y_pred)**2),5)
    RMSE = round(sqrt(mean_squared_error(Y_test, Y_pred)) if custom else np.sqrt(MSE),5)
    MAPE = round(sqrt(mean_absolute_percentage_error(Y_test, Y_pred)) if custom else np.mean(np.abs((Y_test-Y_pred)/(Y_test))),5)
    Adjusted_rand = round(adjusted_rand_score(Y_test, Y_pred),5)
    R2 = round(1-(MSE/(np.mean((Y_test-np.mean(Y_test))**2))),5)
    print(pd.DataFrame([MAE, MSE, RMSE, MAPE, Adjusted_rand, R2], index = ['MAE', 'MSE', 'RMSE', 'MAPE', 'Adjusted_rand', 'R^2'], columns=['Метрики качества']));

def regression_metrics(y_test, y_pred, within_percantage = 10):
    MAE = round(mean_absolute_error(y_test, y_pred), 5)
    RMSE = round(sqrt(mean_squared_error(y_test, y_pred)), 5)
    R2 = round(r2_score(y_test, y_pred),5 )
    WP = round(within_percantage_metric(y_test, y_pred, within_percantage), 5)
    print(pd.DataFrame([MAE, RMSE, R2, WP], index=['MAE', 'RMSE', 'R2', 'WP'], columns=['Метрики качества']));

def within_percantage_metric(y_true : pd.Series, y_pred : pd.Series, percantage = 10 ) -> float:
    errors = np.abs((y_true - y_pred)/y_true) * 100
    return np.mean(errors <= percantage) * 100
    