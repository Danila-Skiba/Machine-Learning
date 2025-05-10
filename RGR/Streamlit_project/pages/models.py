import folium
import numpy as np
import pandas as pd
import streamlit as st
from components.preprocessing import get_test_samples
from components.visualisation import show_model_metrics, plot_map_accuracy_model, plot_distplot_regressors
from components.get_data_user import get_data, plot_sidebar_map, get_test_vector
from components.load import load_config
from models_ml.models import get_models

from streamlit_folium import folium_static

#Load config
config = load_config("model_readme.toml")
config_models = load_config("models_info.toml")

models = get_models()

x_test, y_test = get_test_samples()


st.title(config['app']['title'])

st.sidebar.header(config['sidebar']['models_name'])

select_model = st.sidebar.selectbox("Выберите модель", [key for key in models.keys()])

#Model info
name_eng = models[select_model]['name_eng']
st.header(name_eng)

with st.expander(f"Больше информации о {name_eng}"):
    st.write(models[select_model]['info'])

if "images_path" in  models[select_model]:
    st.image(image=models[select_model]['images_path'])

#Model learn code
st.header("Обучение модели")
st.code(body=models[select_model]['learn_code'])

#Model metrics
st.header("Метрики модели")
model = models[select_model]['model']
model_metrics = models[select_model]['metrics']
y_pred = model.predict(x_test)
show_model_metrics(model_metrics)

with st.expander("**Больше информации о метриках**"):
    st.write(config_models['metrics']['help'])

#Model visualisation
st.header("Точность модели")

col1, col2 = st.columns(2)
with st.expander("**Подробнее о карте**"):
    st.markdown(config['mapa']['description'])
with col1:
    show_map = st.button("Показать карту")
with col2:
    hide_map = st.button("Скрыть карту")
if show_map:
    map = plot_map_accuracy_model(x_test, y_test, y_pred)
    folium_static(map)
elif hide_map:
    map = None
st.divider()

with st.expander('Подробнее о графике'):
    st.write(config['graphic']['distplot'])
dist_plot_model = plot_distplot_regressors(regressors={f'{name_eng}': model}, 
X_test=x_test, y_test=y_test,x_title= 'Price', y_title="Плотность", nameplot=config['graphic']['name'])
st.plotly_chart(dist_plot_model)

#Prediction
st.sidebar.header("Заполнение данных", help=config['helps']['sidebar_help'])
st.sidebar.subheader("Локация на карте")
if 'latitude' not in st.session_state:
    st.session_state.latitude = None
if 'longitude' not in st.session_state:
    st.session_state.longitude = None
clicked_lat, clicked_lon = plot_sidebar_map()
st.sidebar.divider()
values = get_data()
y_user_pred = None
try:
    x_vector_test = get_test_vector(values, clicked_lat, clicked_lon)
    y_user_pred = model.predict(x_vector_test)[0]
    y_user_pred = y_user_pred if y_user_pred >0 else model_metrics['MAE']
except:
    st.sidebar.error("Введите верные параметры")
st.sidebar.divider()
predict = st.sidebar.button("Предсказать стоимость")
if predict and y_user_pred:
    st.sidebar.header(f"Стоимость: ```{y_user_pred:.4f}``` млн рупий")