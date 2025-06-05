import plotly as pt
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_char_epochs_metrics(epochs, train_loss, train_metric, name= "", metric_name = "Метрика"):
    fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=(f"Функция потерь от количества эпох", "Значение метрики от количества эпох"),
                    specs=[[{"type": "scatter"}, {"type": "scatter"}]])


    fig.add_trace(
    go.Scatter(x=epochs, y=train_loss, mode='lines', name='Функция потерь',
               line=dict(color="#CA8BDF", width=2)),
    row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=epochs, y=train_metric, mode='lines', name= metric_name,
               line=dict(color="#8DD5C3", width=2)),
     row=1, col=2
    )

    fig.update_layout(
        title_text=f"Метрики обучения модели {name}",
        showlegend=True,
    )

    fig.show()