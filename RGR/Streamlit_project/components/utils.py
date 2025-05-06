import streamlit as st
import pandas as pd
import io

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

@st.cache_data
def load_default_data(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
        return None 