from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error


def get_metrics(y_true, y_pred):
    metrics = {
        "MSE": round(mean_squared_error(y_true, y_pred), 3),
        'MAE': round(mean_absolute_error(y_true, y_pred),3),
        "MAPE": round(mean_absolute_percentage_error(y_true, y_pred),3) ,
        "RMSE": round(root_mean_squared_error(y_true, y_pred),3),
        "R2": round(r2_score(y_true, y_pred),3)
    }

    return metrics 