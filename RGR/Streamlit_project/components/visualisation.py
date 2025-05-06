import streamlit as st
import plotly.express as px
import io
import matplotlib.pyplot as plt
import missingno as msno
import folium


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
def plot_map(data):
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