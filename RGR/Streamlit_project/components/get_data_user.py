import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


def plot_sidebar_map():

    borders = [
    (19.5, 72.7),
    (18.8, 72.7),
    (18.8, 73.2),
    (19.5, 73.2)
    ]

    def is_within_border(lat, lon, borders):
        lat_min = min(b[0] for b in borders)
        lat_max = max(b[0] for b in borders)
        lon_min = min(b[1] for b in borders)
        lon_max = max(b[1] for b in borders)
        return lat_min<=lat <=lat_max and lon_min<=lon<=lon_max

    center_lat = (min(b[0] for b in borders)+ max(b[0] for b in borders))/2
    center_lon = (min(b[1] for b in borders)+ max(b[1] for b in borders))/2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

   

    folium.Polygon(
        locations=borders,
        color = "red",
        weight = 2,
        fill = True,
        fill_opacity = 0.2
    ).add_to(m)

    folium.LatLngPopup().add_to(m)

    with st.sidebar:
        with st.sidebar.container(height=300):
            st.write("Выберите точку на карте:")
            map_data = st_folium(m, width=400, height=300)
            st.markdown("""
    <style>
    .map_data {  /* Замените на реальный класс, если отличается */
    margin-bottom: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if  map_data and map_data['last_clicked']:
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lon = map_data['last_clicked']['lng']

        if is_within_border(clicked_lat, clicked_lon, borders):
            st.session_state.latitude = clicked_lat
            st.session_state.longitude = clicked_lon
            st.sidebar.success(f"Точка выбрана: latitude = {clicked_lat:.4f}, longitude = {clicked_lon:.4f}")
            return clicked_lat, clicked_lon
        else:
            st.sidebar.error("Выбранная точка находится за пределами допустимой области!")
 
    return None, None
    


def get_data():

    st.sidebar.subheader("Площадь жилья", help="кв.м")
    select_area = st.sidebar.slider("```area```", 500, 8000)

    st.sidebar.subheader("Количество балконов")
    select_balcony = st.sidebar.slider("```balcony```",0,20)

    st.sidebar.subheader("Количество спальнь")
    select_bedrooms = st.sidebar.slider("```bedrooms```",0,20)

    st.sidebar.subheader("Количество душевых")
    select_bathrooms = st.sidebar.slider("```bathrooms```",0,20)

    st.sidebar.subheader("Готов для заселения?")
    select_ready_to_move = st.sidebar.selectbox("```ready_to_move```", ["Да", "Нет"])
    ready_to_move = 1 if select_ready_to_move == "Да" else 0

    st.sidebar.subheader("Статус жилья")
    select_new_or_old = st.sidebar.selectbox("```new_or_old```", ['Новое', "Вторичное"])
    new_or_old =  1 if select_new_or_old == "Новое" else 0

    st.sidebar.subheader("Количество парквочных мест")
    select_parking = st.sidebar.slider("```parking```",0,10)

    st.sidebar.subheader("Количество лифтов")
    select_lift = st.sidebar.slider("```lift```", 0,10)

    st.sidebar.subheader("Тип жилья")
    select_flat_or_individ = st.sidebar.selectbox("```flat_or_individual```", ["Квартира", "Частный дом"])
    flat_or_individ = 1 if select_flat_or_individ == "Квартира" else 0

    st.sidebar.subheader("Статус обустройства жилья", help="Обстановка мебелью/бытовой техникой")
    select_furn_status = st.sidebar.selectbox("```furnished_status```", ['Обустроено', "Не обустроено"])

    furn_status_semi = 1 if select_furn_status == "Обустроено" else 0
    furn_status_unfurn = 1 if select_furn_status == "Не обустроено" else 0

    return select_area, select_bedrooms, select_bathrooms,select_balcony, ready_to_move, new_or_old, select_parking, select_lift, flat_or_individ, furn_status_semi, furn_status_unfurn


def get_test_vector(values, click_lat, click_lon):
    x_vector_test = pd.DataFrame({
        "area": [values[0]],
        'latitude': [click_lat],
        "longitude": [click_lon],
        'bedrooms': [values[1]],
        'bathrooms': [values[2]],
        'balcony': [values[3]],
        'ready_to_move': [values[4]],
        'new_housing': [values[5]],
        'parking': [values[6]],
        'lift': [values[7]],
        'flat_or_individual': [values[8]],
        'furnished_status_Semi-Furnished': [values[9]],
        'furnished_status_Unfurnished': [values[10]]
    })

    return x_vector_test