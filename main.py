import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier


def estadisticas():
    URL = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
    dfIris = pd.read_csv(URL)

    st.title("Analisis estadistico Iris Dataset")

    components.html("""<hr style="height:3px;border:none;color:#333"/>""")

    st.dataframe(dfIris.head())
    st.header("Estadísticas")
    st.write("Filas, columnas:")
    st.write(dfIris.shape)

    st.write("Describe:")
    st.dataframe(dfIris.describe())

    st.write("Clases:")
    st.write(dfIris["variety"].value_counts())

def graficas():
    URL = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
    dfIris = pd.read_csv(URL)

    st.title("Visualización")

    st.subheader(dfIris.columns[0])
    fig = px.box(dfIris, y=dfIris.columns[0])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(dfIris.columns[1])
    fig = px.box(dfIris, y=dfIris.columns[1])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(dfIris.columns[2])
    fig = px.box(dfIris, y=dfIris.columns[2])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(dfIris.columns[3])
    fig = px.box(dfIris, y=dfIris.columns[3])
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(dfIris, y="variety")
    #dfIris.plot(kind='box', )
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("Histogramas")
    for i in range(0, len(dfIris.columns)):
        fig = px.histogram(dfIris, x=dfIris.columns[i])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Gráfica de correlación")
    fig = px.scatter_matrix(dfIris,
    dimensions=dfIris.columns[0:4],
    color="variety")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlación - Mapa de color")
    df_corr=dfIris.corr()
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
        x= df_corr.columns,
        y= df_corr.index,
        z= np.array(df_corr)
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def modelos():
    url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
    name = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    st.title("Modelo")
    dfIris = pd.read_csv(url, names=name)
    dfIris = dfIris[dfIris["sepal-length"] != "sepal.length"]
    # dividir los datos en entrenamiento y prueba

    x_train, x_test, y_train, y_test = train_test_split(dfIris[dfIris.columns[0:4]], dfIris[dfIris.columns[-1]], test_size=0.2)

    st.subheader("Shape X_train:")
    st.write(x_train.shape)
    st.subheader("Shape X_test:")
    st.write(x_test.shape)

    modelos = []

    modelo = LogisticRegression(random_state=0).fit(x_train, y_train)
    modelo.score(x_test, y_test)
    modelo.predict(x_test)

    # kfold = StratifiedF old(n_splits=10, random_state=1)

    #modeloKN = KNeighborsClassifier(n_neighbors=3)

    #modeloKN.fit(x_train, y_train)
    #modeloKN.score(x_test)

    st.subheader("Precision:")
    st.write(modelo.score(x_test, y_test))

    st.subheader("Predicción:")
    st.write(modelo.predict(x_test))

opciones = ["Estadisticas", "Graficas", "Modelos"]
selection = st.sidebar.selectbox("Paginas:", opciones, index=0)
if selection == "Estadisticas":
    estadisticas()
elif selection == "Graficas":
    graficas()
else:
    modelos()


