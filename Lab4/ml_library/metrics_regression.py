from numpy import sqrt
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


def print_error(Y_test, Y_pred, custom = False):
    MAE = round(mean_absolute_error(Y_test, Y_pred) if custom else np.mean(np.abs(Y_test-Y_pred)), 5)
    MSE = round(mean_squared_error(Y_test, Y_pred) if custom else np.mean((Y_test-Y_pred)**2),5)
    RMSE = round(sqrt(mean_squared_error(Y_test, Y_pred)) if custom else np.sqrt(MSE),5)
    MAPE = round(sqrt(mean_absolute_percentage_error(Y_test, Y_pred)) if custom else np.mean(np.abs((Y_test-Y_pred)/(Y_test))),5)
    Adjusted_rand = round(adjusted_rand_score(Y_test, Y_pred),5)
    R2 = round(1-(MSE/(np.mean((Y_test-np.mean(Y_test))**2))),5)
    print(pd.DataFrame([MAE, MSE, RMSE, MAPE, Adjusted_rand, R2], index = ['MAE', 'MSE', 'RMSE', 'MAPE', 'Adjusted_rand', 'R^2'], columns=['Метрики качества']));
