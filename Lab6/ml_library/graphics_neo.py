import matplotlib.pyplot as plt
import plotly.graph_objects as go

def count_TP_TN_FP_FN(y_true, y_pred):
    TP = (y_true == 1) & (y_pred == 1)
    TN = (y_true == 0) & (y_pred == 0)
    FP = (y_true == 0) & (y_pred == 1)
    FN = (y_true == 1) & (y_pred == 0)
    return TP,TN,FP,FN

def plot_miss_distance_vs_velocity(X_test, y_true, y_pred):
    miss_distances = X_test[:, 3] 
    velocities = X_test[:, 2]  


    colors = {'TP': '#2ca02c', 'TN': '#1f77b4', 'FP': '#ff7f0e', 'FN': '#d62728'}
    labels = {0: 'Безопасный', 1: 'Опасный'}

   
    TP,TN,FP,FN = count_TP_TN_FP_FN(y_true, y_pred)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=miss_distances[TP], y=velocities[TP],
        mode='markers', marker=dict(color=colors['TP'], symbol='circle', opacity=0.7),
        name=f'TP (Опасный - {labels[1]})'
    ))
    

    fig.add_trace(go.Scatter(
        x=miss_distances[TN], y=velocities[TN],
        mode='markers', marker=dict(color=colors['TN'], symbol='circle', opacity=0.7),
        name=f'TN (Безопасный - {labels[0]})'
    ))

  
    fig.add_trace(go.Scatter(
        x=miss_distances[FP], y=velocities[FP],
        mode='markers', marker=dict(color=colors['FP'], symbol='x', opacity=0.7),
        name='FP (Ложная тревога)'
    ))
    fig.add_trace(go.Scatter(
        x=miss_distances[FN], y=velocities[FN],
        mode='markers', marker=dict(color=colors['FN'], symbol='x', opacity=0.7),
        name='FN (Пропущенная угроза!)'
    ))

    fig.update_layout(
        title='Распределение астероидов по расстоянию промаха и скорости',
        xaxis_title='Расстояние промаха от земли (miss_distance)',
        yaxis_title='Относительная скорость (relative_velocity)',
        legend_title='Категории'
    )

    fig.show()

def plot_magnitude_vs_velocity(X_test, y_true, y_pred, filename="magnitude_vs_velocity"):
    magnitudes = X_test[:, 4]
    velocities = X_test[:, 2]
    fig = go.Figure()


    colors = {'TP': '#2ca02c', 'TN': '#1f77b4', 'FP': '#ff7f0e', 'FN': '#d62728'}
    labels = {0: 'Безопасный', 1: 'Опасный'}

    TP,TN,FP,FN = count_TP_TN_FP_FN(y_true, y_pred)


    fig.add_trace(go.Scatter(
        x=magnitudes[TP], y=velocities[TP],
        mode='markers', marker=dict(color=colors['TP'], symbol='circle', opacity=0.7),
        name=f'TP (Опасный - {labels[1]})'
    ))
    fig.add_trace(go.Scatter(
        x=magnitudes[TN], y=velocities[TN],
        mode='markers', marker=dict(color=colors['TN'], symbol='circle', opacity=0.7),
        name=f'TN (Безопасный - {labels[0]})'
    ))
    fig.add_trace(go.Scatter(
        x=magnitudes[FP], y=velocities[FP],
        mode='markers', marker=dict(color=colors['FP'], symbol='x', opacity=0.7),
        name='FP (Ложная тревога)'
    ))
    fig.add_trace(go.Scatter(
        x=magnitudes[FN], y=velocities[FN],
        mode='markers', marker=dict(color=colors['FN'], symbol='x', opacity=0.7),
        name='FN (Пропущенная угроза!)'
    ))
    

    fig.update_layout(
        title='Зависимость классификации от absolute_magnitude и relative_velocity',
        xaxis_title='absolute_magnitude',
        yaxis_title='relative_velocity',
        legend_title='Категории'
    )
    

    fig.show()