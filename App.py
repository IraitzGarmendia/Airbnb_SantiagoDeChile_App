#-----------------------------LIBRERIAS-----------------------------
import streamlit as st
import matplotlib as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pycaret.regression import load_model, predict_model

model = load_model('ml(SC)_airbnb')
st.title("Sistema de prediccion de precios en Santiago de Chile")

neighbourhood = st.selectbox('Barrio', options=['Providencia', 'Lo Barnechea', 'Recoleta', 'Santiago',
       'La Florida', 'Las Condes', 'La Reina', 'Ñuñoa', 'Independencia',
       'San Miguel', 'Peñalolén', 'Vitacura', 'Maipú', 'Estación Central',
       'Pedro Aguirre Cerda', 'San Joaquín', 'Macul', 'El Bosque',
       'Lo Espejo', 'La Cisterna', 'Quinta Normal', 'Quilicura',
       'Pudahuel', 'Lo Prado', 'Huechuraba', 'Renca', 'Cerrillos',
       'La Granja', 'Conchalí', 'Cerro Navia', 'San Ramón'])

property_type = st.selectbox('Tipo de Propiedad', options=['Private room in rental unit', 'Entire rental unit',
       'Private room in home', 'Entire condo', 'Private room in bungalow',
       'Shared room in home', 'Private room in condo',
       'Entire serviced apartment', 'Entire villa', 'Casa particular',
       'Private room in bed and breakfast', 'Entire loft',
       'Room in serviced apartment', 'Entire home',
       'Private room in tower', 'Shared room in rental unit',
       'Entire vacation home', 'Private room',
       'Private room in guest suite', 'Private room in guesthouse',
       'Shared room in hostel', 'Private room in townhouse',
       'Shared room in loft', 'Tiny home', 'Private room in hostel',
       'Entire townhouse', 'Entire cabin', 'Entire place',
       'Room in boutique hotel', 'Private room in serviced apartment',
       'Private room in cabin', 'Entire guesthouse',
       'Private room in farm stay', 'Entire bungalow',
       'Entire guest suite', 'Private room in casa particular',
       'Private room in loft', 'Private room in boat', 'Entire chalet',
       'Shared room in bed and breakfast', 'Tower',
       'Shared room in guest suite', 'Private room in nature lodge',
       'Private room in chalet', 'Shared room in condo',
       'Shared room in guesthouse', 'Private room in villa',
       'Room in aparthotel', 'Private room in houseboat',
       'Shared room in houseboat', 'Entire cottage', 'Shared room',
       'Private room in cottage', 'Private room in vacation home',
       'Room in hostel', 'Private room in floor','Private room in pension', 'Farm stay', 'Ranch',
       'Private room in castle', 'Yurt', 'Shipping container',
       'Private room in dome', 'Private room in treehouse', 'Windmill',
       'Shared room in hotel', 'Private room in tent', 'Castle', 'Cave',
       'Earthen home', 'Dome', 'Private room in yurt'])

accommodates = st.slider('Número de Personas', min_value=1, max_value=17, value=1)
room_type = st.selectbox('Tipo de Habitación', options=['Private room', 'Entire home/apt', 'Shared room'])
maximum_nights = st.slider('Noches Máximas', min_value=1, max_value=100, value=1)
minimum_nights = st.slider('Noches Mínimas', min_value=1, max_value=10, value=1)

input_data = pd.DataFrame([[
    neighbourhood, property_type, accommodates, room_type,
    maximum_nights, minimum_nights
]], columns=['neighbourhood', 'property_type', 'accommodates', 'room_type', 'maximum_nights', 'minimum_nights'])


if st.button('¡Descubre el precio!'):
    prediction = predict_model(model, data=input_data)
    st.write(str(prediction["prediction_label"].values[0]) + ' euros')