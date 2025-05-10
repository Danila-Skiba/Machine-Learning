import streamlit as st
import pandas as pd
from streamlit_folium import folium_static

from components.visualisation import show_dataframe, plot_scatter, plot_hot_map, plot_histograms, plot_msno, get_cached_map
from components.utils import load_default_data, download_data
from components.preprocessing import handle_missing_values, encoding_data
from components.load import load_config

config = load_config("eda_readme.toml")

st.markdown(config['app']['title'])

with st.expander(
    config['app']['page_info'],expanded=False
):
    st.markdown(config['links']['start_dataset'])
    

if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None

#Load data

st.sidebar.title("1. Загрузка данных")
with st.sidebar:
   if st.button("Загрузить данные"):
      
       df = load_default_data("RGR/Streamlit_project/src/mumbai_regression.csv")
       st.session_state.df = df

       if df is not None:
           st.success("Стартовый датасет загружен")
       else:
           st.error("Не удалось загрузить стартовый датасет")

#Preprocessing

st.sidebar.header("2. Настройки предобработки")

#Handle missing
st.sidebar.subheader("Обработка пропусков")
if 'df' in st.session_state:
    df = st.session_state.df
    if df.isnull().sum().sum() > 0:
        handle_missing = st.sidebar.checkbox("Обработать пропуски", help=config["helps"]["handle_missing"])
        if handle_missing:
            imputation_method = st.sidebar.selectbox("Метод заполнения числовых признаков", 
            ["mean", "median", "most_frequent"], 
            help=config["helps"]["imput_method"])
    else:
        st.sidebar.info("Пропуски в данных отсутствуют.")

    #Encode category features
    st.sidebar.subheader("Кодирование категориальных переменных")
    categorical = [col for col in df.columns if df[col].dtype in ["object", "category"]]
    if categorical:
        st.sidebar.write(f"Категориальные столбцы: ```{', '.join(categorical)}```")
        encode_categorical = st.sidebar.checkbox("Кодировать категориальные переменные", 
            help=config['helps']['encode'])
        if encode_categorical:
            encoding_method = st.sidebar.selectbox("Метод кодирования", ["one-hot", "label"], 
            help=config['helps']['encode_method'])
    else:
        st.sidebar.info("Категориальные признаки отсутствуют")
    
    #Select features 
    st.sidebar.subheader("Выбор признаков")
    selected_cols = st.sidebar.multiselect("Выберите столбцы для включения в обработанный датасет", 
                    df.columns, default=df.columns.to_list(), 
                    help=config['helps']['select_cols'])
    
    #Start preprocessing

    if st.sidebar.button("Применить предобработку"):
        try:
            df_processed = df.copy()
            if selected_cols: 
                df_processed = df_processed[selected_cols]
            else:
                st.error("Выберите хотя бы один столбец")
                st.stop()
            if handle_missing:
                df_processed = handle_missing_values(df_processed, imputation_method)
            if encode_categorical:
                df_processed = encoding_data(df_processed, encoding_method)

            st.session_state.df_processed = df_processed
            st.success("Предобработка завершена!")
            st.balloons()
        except Exception as e:
            st.error(f"Ошибка при выполнении: {e}")


    #Show start data

    df = st.session_state.df
    st.divider()
    st.subheader("Исходные данные")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Данные", "Описание данных", 
    "Жильё на карте", "Тепловая карта", "Карта пропусков"])

    with tab1:
        show_dataframe(df)
        with st.expander("**Статистика**", expanded=False):
            st.write(df.describe())
    with tab2: 
        st.markdown(config['start_df']['describe_features'])
        
    with tab3:
       show_map = st.button("Показать карту")
       if show_map:
            mapa = get_cached_map(df)
            folium_static(mapa)
       mapa = None
    with tab4:
        with st.expander("### **Корреляция числовых признаков**"):
            st.write(config['diagrams']['hot_map'])
        plot_hot_map(df)

    with tab5:
        plot_msno(df)

    #Data analysis

    with st.expander("**Анализ данных**"):
        st.write(config['start_df']['data_analysis'])
        
    with st.expander("**Предположения и выводы**"):
        st.write(config['start_df']['reports'])
    st.divider()

    #Data preprocessing

    st.markdown("### Предобработанные данные")
    my_data = load_default_data("RGR/Streamlit_project/src/result_mumbai.csv")
    tab1, tab2, tab3, tab4 = st.tabs(['Данные','Тепловая карта',
    "Гистограммы",'Диаграмма рассеивания' ])
    with tab1:
        show_dataframe(my_data, True, key="download_my_data")
    with tab2:
        with st.expander("### **Корреляция числовых признаков**"):
            st.write(config['diagrams']['hot_map'])
        plot_hot_map(my_data)
    with tab3: 
        plot_histograms(my_data)
    with tab4:
        with st.expander("**Диаграмма рассеивания**"):
            st.write(config['diagrams']['scatter'])
        plot_scatter(my_data)
    st.divider()

#Data preprocessing with user/download data
        
if st.session_state.df_processed is not None:
    df_new = st.session_state.df_processed
    
    st.subheader("Ваши обработанные данные")
    tab1, tab2, tab3 = st.tabs(['Данные',"Гистограмма", "Карта пропусков"])
    with tab1:
        show_dataframe(df_new)
        download_data(df_new, "download_df_new")
        with st.expander("**Статистика**"):
            st.write(df_new.describe())
    with tab2:
        plot_histograms(df_new)
    with tab3:
        plot_msno(df_new)