import streamlit as st
nav = [
    st.Page("pages/eda.py", title="Предобработка данных", icon="📊"),
    st.Page("pages/author_info.py", title="Информация об авторе", icon="🤠"),
    st.Page("pages/models.py", title="Модели", icon="💡"),
    st.Page("pages/about.py", title="О сайте", icon="❔"),
]
pg = st.navigation(nav)
pg.run()
