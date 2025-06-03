import numpy as np
from sklearn.model_selection import cross_val_score


def k_fold(model, X_train, y_train, scoring = 'accuracy', print_result = True):
    scores  = cross_val_score(model, X_train,y_train, cv = 5, scoring=scoring)
    if print_result:
        print("Оценки кросс-валидации на обучающем наборе", scores)
        print("Средняя оценка кросс-валидации на обучающем наборе", np.mean(scores))
    return np.mean(scores)