from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, r2_score
import seaborn as sns
from scipy.stats import gaussian_kde

def ROC_curve(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr,tresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go

def plot_roc_curves(classifiers, X_test, y_test, title='ROC-Кривые для различных классификаторов'):
    fig = go.Figure()
    
    for name, clf in classifiers.items():
        y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{name} (AUC = {roc_auc:.2f})',
            line=dict(color=f'#{hash(name) % 16777215:06x}', width=2)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Случайный (AUC = 0.5)',
        line=dict(color='gray', dash='dash', width=2)
    ))
    

    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        legend_title='Классификаторы',
        hovermode='x unified'
    )
    
    return fig


def print_chart(name, Y_test, Y_pred):
    plt.figure(figsize=(6,4))
    plt.title(name)
    sns.distplot(Y_test, hist=False, color='Red', label='Actual values')
    sns.distplot(Y_pred, hist=False, color='Green', label='Predicted values')
    plt.legend()
    plt.show();

def plot_distplot_regressors(regressors, X_test, y_test,x_title,  y_title, nameplot = 'Графики предсказаний для регрессеров'):
    fig = go.Figure()
    kde_test = gaussian_kde(y_test)
    x_true_range = np.linspace(min(y_test), max(y_test), 200)
    fig.add_trace(go.Scatter(
        x=x_true_range,
        y = kde_test(x_true_range),
        mode = 'lines',
        name = "Actual values",
        line = dict(color='red')
    ))
    for name, regressor in regressors.items():
        y_pred = regressor.predict(X_test)
        x_range = np.linspace(min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred)), 200)
        kde_pred = gaussian_kde(y_pred)

        fig.add_trace(go.Scatter(
            x=x_range,
            y = kde_pred(x_range),
            mode='lines',
            name = f"{name}, {r2_score(y_test, y_pred):.2f}",
            line=dict(color=f'#{hash(name) % 16777215:06x}', width=2)
        ))
    fig.update_layout(
        title=nameplot,
        xaxis_title = x_title,
        yaxis_title=y_title,
        legend_title='Regressors',
        hovermode='x unified'
    )

    return fig 
        

    

