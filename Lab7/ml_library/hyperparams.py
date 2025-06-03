


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt

def formated_params(items):
    formatted_params = []
    output = ""
    for key, value in items.items():
        if isinstance(value, str):
            formatted_params.append(f"'{key}':'{value}'")  
        elif isinstance(value, float):
            formatted_params.append(f"'{key}': {round(float(value), 3)}")
        else:
            formatted_params.append(f"'{key}':'{value}'")
    output += ", ".join(formatted_params)
    return output

def gridSearchCV_params(model, parameters,X_train, Y_train):
    optimal = GridSearchCV(model, parameters).fit(X_train, Y_train)
    output = formated_params(optimal.best_params_)
    print(f" Лучшие параметры для модели (GridSearchCV) {output}")
    return optimal.best_estimator_

def randomizedSearchCV_params(model, parameters, X_train, Y_train):
    optimal = RandomizedSearchCV(model, parameters).fit(X_train, Y_train)
    output = formated_params(optimal.best_params_)
    print(f"Лучшие параметры для модели (RandomizedSearchCV) {output}")
    return optimal.best_estimator_


import optuna

def optuna_params(objective, aim, n_trials= 100):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=aim)
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    return best_params