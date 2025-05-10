import streamlit as st
# from st_pages import get_nav_from_toml
# st.set_page_config(page_icon="🌟")
# nav = get_nav_from_toml("config/pages.toml")
# pg = st.navigation(nav)
# pg.run()
nav = [
    st.Page("pages/eda.py", title="Предобработка данных", icon="📊"),
    st.Page("pages/author_info.py", title="Информация об авторе", icon="🤠"),
    st.Page("pages/models.py", title="Модели", icon="💡"),
    st.Page("pages/about.py", title="О сайте", icon="❔"),
]
pg = st.navigation(nav)
pg.run()