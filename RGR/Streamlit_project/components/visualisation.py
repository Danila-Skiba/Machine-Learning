from math import sqrt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,  r2_score
import streamlit as st
import plotly.express as px
import io
import matplotlib.pyplot as plt
import missingno as msno
import folium
import plotly.graph_objects as go

from streamlit_folium import st_folium 
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error


def plot_scatter(data):
    plot_col_y = st.selectbox("Выберите целевой признак ", data.columns)
    plot_col_x = st.selectbox("Выберите косвенный признак", data.columns)
    fig = px.scatter(
        data, x =plot_col_x,
        y = plot_col_y,
        title= f"Диаграмма рассеивания {plot_col_y} от {plot_col_x}"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_dataframe(data, download = False, key= None):
    with st.expander("**DataFrame**" ,expanded=True):
        st.dataframe(data)
        memory_usage = data.memory_usage(deep = True).sum()
        memory_usage_kb = memory_usage/ 1024
        st.write(f"```{memory_usage_kb:.3f} KB```", unsafe_allow_html=True)
        if download:
            download_data(data, key)

def download_data(data, key):
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    download = st.download_button(
    label="Скачать CSV",
    data=csv_data,
    file_name="data.csv",
    mime="text/csv",
    key=key)

def get_numerical_cols(data):
    return data.select_dtypes(include=['number']).columns.tolist()

def plot_hot_map(data):
    numeric_cols = get_numerical_cols(data)
    if numeric_cols:
        corr_matrix = data[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True)
        fig.update_layout(
            width=1000,
            height=800,
            font=dict(size=12), 
            margin=dict(l=0, r=0, b=0, t=0, pad=2) 
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_msno(data):
    with st.expander("### **Карта пропусков**"):
        st.write("""показывает объём пропущенных значений у признака""")
    fig, ax  = plt.subplots(figsize = (10,6))
    msno.matrix(data, ax=ax)
    st.pyplot(fig)

def plot_histograms(data):
    st.write("### Гистограмма распредления признаков")
    plot_col = st.selectbox("Выберите признак для гистограммы", data.columns)
    fig = px.histogram(data, x= plot_col, title=f"Распределение {plot_col}", nbins=100)
    st.plotly_chart(fig, use_container_width=True)

def color_producer(price, data):
    if price <= data['price'].quantile(0.25):
        return '#1a9850' 
    elif price <= data['price'].quantile(0.5):
        return '#91cf60'  
    elif price <= data['price'].quantile(0.75):
        return '#fee08b' 
    else:
        return '#d73027'
    
@st.cache_data
def plot_map(data,model_accuracy = False):
    mumbai_lat = data['latitude'].mean()
    mumbai_lon = data['longitude'].mean()

    m = folium.Map(location=[mumbai_lat, mumbai_lon], zoom_start=11)

    for index, row in data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3, 
            color=color_producer(row['price'], data),
            fill=True,
            fill_opacity=0.7,
            popup=f"Цена: {row['price']}"  
        ).add_to(m)
    return m


def plot_map_accuracy_model(x_test, y_test, y_pred):
    mumbai_lat = x_test['latitude'].mean()
    mumbai_lon = x_test['longitude'].mean()

    y_test = np.array(y_test)
    y_pred =np.array(y_pred)

    m = folium.Map(location=[mumbai_lat, mumbai_lon], zoom_start=11)
    i =0
    for _, row in x_test.iterrows():
        acc =  abs(y_test[i]-  y_pred[i]) < mean_absolute_error(y_test, y_pred)
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color = '#91cf60'  if acc else '#d73027',
            fill= True,
            fill_opacity = 0.7,
            popup= f"Цена: {y_test[i]}" if acc else f"True: {y_test[i]}, Predict: {y_pred[i]}"
        ).add_to(m)
        i+=1
    return m


@st.cache_resource
def get_cached_map(data):
    return plot_map(data)


def markdown_style(metric):
    return f"font-weight: bold; font-size: 20px;'>{metric}</p>"


def show_model_metrics(metrics):
  

    col1, col2, col3, col4, col5 = st.columns(5)
    style = f"<p style='color: #ff335c ;"
    col1.markdown(
            f"{style}"
            f"{markdown_style("MAE")}",
            unsafe_allow_html=True,
    )
    col1.write(f"**{metrics['MAE']}**")
    col2.markdown(
            f"{style}"
            f"{markdown_style("MSE")}",
            unsafe_allow_html=True,
    )
    col2.write(f"**{metrics['MSE']}**")
    col3.markdown(
            f"{style}"
            f"{markdown_style("RMSE")}",
            unsafe_allow_html=True,
    )
    col3.write(f"**{metrics['RMSE']}**")
    col4.markdown(
            f"{style}"
            f"{markdown_style("MAPE")}",
            unsafe_allow_html=True,
    )
    col4.write(f"**{metrics['MAPE']}**")
    col5.markdown(
            f"{style}"
            f"{markdown_style("R2")}",
            unsafe_allow_html=True,
    )
    col5.write(f"**{metrics['R2']}**")


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
            name = f"{name}",
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



    