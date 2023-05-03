import pandas as pd
import streamlit as st
import streamlit.components.v1 as componets

URL="https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
dfIris= pd.read_csv(URL)

st.title("Analisis estadistico Iris Dataset")

componets.html("""<hr style="height:3px;border:none;color:#333"/hr>""")

st.dataframe(dfIris.head())
st.header("Estadísticas")
st.write("Filas, columnas:")
st.write(dfIris.shape)

st.write("Describe:")
st.dataframe(dfIris.describe())

st.write("Clases:")
st.write(dfIris["variety"].value_counts())