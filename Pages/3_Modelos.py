import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
name = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dfIris = pd.read_csv(url, names=name)

#dividir los datos en entrenamiento y prueba


x_train, x_test, y_train, y_test = train_test_split(dfIris[dfIris.columns[0:4]], dfIris[dfIris.columns[-1]], test_size=0.2)
x_test = x_test[x_test["sepal-length"]!="sepal-length"]

print(x_train.shape)
print(x_test.shape)

modelos = []

modelo = LogisticRegression(random_state=0).fit(x_train, y_train)
modelo.score(x_test, y_test)
modelo.predict(x_test)

# kfold = StratifiedF old(n_splits=10, random_state=1)

#modeloKN = KNeighborsClassifier(n_neighbors=3)

#modeloKN.fit(x_train, y_train)
#modeloKN.score(x_test)