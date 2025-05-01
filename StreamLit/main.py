import streamlit as st

# Определяем страницы
page1 = st.Page("pages/page1.py", url_path="page-one")
page2 = st.Page("pages/page2.py", url_path="page-two")

# Настраиваем навигацию
st.navigation([page1, page2])

# Контент для главной страницы (опционально)
st.write("Это главная страница")
