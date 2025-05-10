import streamlit as st
st.title("Обо мне")
st.subheader("```ФИО:``` _Скиба Данила Сергеевич_")
st.subheader("```Кафедра:``` Кафедра Хайпа")
st.subheader("```Направление```: 02.03.03 _Математическое обеспечение и администрирование информационных систем_")
st.subheader("```Номер группы```: МО-231")
st.divider()

st.header("Тема Расчётно графической работы")
st.subheader("_Разработка Web-приложения для инференса моделей ML и анализа данных_")
st.sidebar.image(image="components/images/profil2.png", width=400)
show_next = st.sidebar.button("Показать ещё?")
if show_next:
    st.sidebar.image(image="components/images/profil3.png", width=600)

