


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

def optuna_params(objective, aim):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=aim)
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    return best_params


from sklearn.model_selection import cross_val_score

def gen_objective(estimator, grid, X_train, Y_train, metric='r2', **kwargs):
    def objective(trial):
        params = {}
        for name_param, type_param in grid.items():
            if isinstance(type_param, list):
                params[name_param] = trial.suggest_categorical(name_param, type_param)
            if isinstance(type_param, tuple):
                if len(type_param) == 2:
                    params[name_param] = trial.suggest_float(name_param, type_param[0], type_param[1])
                elif len(type_param) == 3:
                    params[name_param] = trial.suggest_float(name_param, type_param[0], type_param[1], log = type_param[2])
        print(params)

        model = estimator(**params, **kwargs)

        score = cross_val_score(model, X_train, Y_train, scoring=metric, cv=5)
        return score.mean()
    return objective

