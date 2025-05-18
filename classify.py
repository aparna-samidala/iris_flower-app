import streamlit as st
from keras.models import load_model 
import numpy as np 
import pandas as pd
from sklearn.datasets import load_iris                                                                                                           


model = load_model("model.h5")
label = np.load("label.npy")

st.title("Welcome to flower prediction app")


a = float(st.number_input("sepal length in cm"))
b = float(st.number_input("sepal width in cm"))
c = float(st.number_input("petal length in cm"))
d = float(st.number_input("petal width in cm"))


if st.button("Predict"):
    prediction = model.predict(np.array([a,b,c,d]).reshape(1,-1))
    pred = label[np.argmax(prediction)]
    st.subheader(pred)
    
    
    if pred == "Iris-setosa":
        st.image("setosa.jpg")
    elif pred == "Iris-versicolor":
        st.image("versicolor.jpg")
    elif pred == "Iris-virginica":
        st.image("virginica.jpg")

    
   