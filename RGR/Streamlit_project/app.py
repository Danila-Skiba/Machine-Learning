import streamlit as st
from st_pages import get_nav_from_toml
st.set_page_config(page_icon="🌟")
nav = get_nav_from_toml("config/pages.toml")
pg = st.navigation(nav)
pg.run()

st.sidebar.write("[Исходный код]()")
