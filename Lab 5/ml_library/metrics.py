import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import numpy as np


def classifier_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    precision_rec = recall_score(y_test, y_pred)
    metrics_df = pd.DataFrame({
        'Метрики качества': [accuracy, precision, precision_rec]
    }, index=["accuracy_score", "precision_score", "recall_score"])
    print(metrics_df, "\n")
    print(classification_report(y_test, y_pred))
    print("Матрица ошибок\n",confusion_matrix(y_test, y_pred))


def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred))) 
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        true_index = np.where(classes == true_label)[0][0]
        pred_index = np.where(classes == pred_label)[0][0]
        cm[true_index, pred_index] += 1

    return cm

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def custom_classifier_metrics(y_test, y_pred):
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    total_samples = len(y_test)
    
    accuracy = (tp + tn) / total_samples
    
    if tp + fp == 0:
        precision = 0
        print("Предупреждение: Нет предсказанных положительных, precision = 0")
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0
        print("Предупреждение: Нет реальных положительных, recall = 0")
    else:
        recall = tp / (tp + fn)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    metrics_df = pd.DataFrame(
        [accuracy, precision, recall, f1_score],
        index=["accuracy_score", "precision_score", "recall_score", "f1_score"],
        columns=["Метрики качества"]
    )
    print(metrics_df.round(4))
    print_confusion_matrix(conf_matrix)


def print_confusion_matrix(cm, class_names=None):
    if cm.shape != (2, 2):
        raise ValueError("Функция поддерживает только матрицы 2x2 для бинарной классификации")
    
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    
    print("Матрица ошибок (Строки: Predicted, Столбцы: True):")
    print("------------------")
    print("             ", class_names[0], "   ", class_names[1])
    print("-----------------------------------")
    print("Predicted ", class_names[0], f" | {cm[0, 0]:>6}   {cm[0, 1]:>6} |")
    print("Predicted ", class_names[1], f" | {cm[1, 0]:>6}   {cm[1, 1]:>6} |")
    print("-----------------------------------")