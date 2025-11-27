import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # Predicción del nivel de condición física ''')
st.image("fitness.jpg", caption="Predicción del estado físico basado en hábitos y datos biométricos.")

st.header('Datos del usuario')

def user_input_features():
  # Entrada
  age = st.number_input('Edad:', min_value=10, max_value=100, value=25, step=1)
  heart_rate = st.number_input('Frecuencia cardiaca:', min_value=40, max_value=200, value=80, step=1)
  blood_pressure = st.number_input('Presión arterial (sistólica):', min_value=80, max_value=200, value=120, step=1)
  sleep_hours = st.number_input('Horas de sueño:', min_value=0.0, max_value=15.0, value=7.0, step=0.5)
  nutrition_quality = st.number_input('Calidad de nutrición (0-10):', min_value=0, max_value=10, value=5, step=1)
  activity_index = st.number_input('Índice de actividad (0-10):', min_value=0, max_value=10, value=5, step=1)
  smokes = st.number_input('¿Fuma? (0 = No, 1 = Sí):', min_value=0, max_value=1, value=0, step=1)
  gender = st.number_input('Género (0 = Mujer, 1 = Hombre):', min_value=0, max_value=1, value=0, step=1)

  user_input_data = {
                     'age': age,
                     'heart_rate': heart_rate,
                     'blood_pressure': blood_pressure,
                     'sleep_hours': sleep_hours,
                     'nutrition_quality': nutrition_quality,
                     'activity_index': activity_index,
                     'smokes': smokes,
                     'gender': gender}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

fitness = pd.read_csv('fitness_data.csv', encoding='latin-1')

X = fitness.drop(columns='is_fit')
Y = fitness['is_fit']

classifier = DecisionTreeClassifier( max_depth=5, criterion='gini', min_samples_leaf=25, max_features=6, random_state=1614175)

classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('No está en condición física óptima')
elif prediction == 1:
  st.write('Sí está en buena condición física')
else:
  st.write('Sin predicción')
