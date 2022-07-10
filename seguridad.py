import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

if __name__ == '__security__':
    """
   Initialize data
   """
    x1 = 2 * np.random.rand(100, 1)
    y = 4 + 3 * x1 + np.random.rand(100, 1)
    print("longitud de los datos es: ", len(x1))
    """
   Modificacion con pandas
   """
    data = {
        'equipos_afectados': x1.flatten(),
        'coste': y.flatten()}
    df = pd.DataFrame(data)
    df.head(10)
    df['equipos_afectados'] = df['equipos_afectados'] * 1000
    df['equipos_afectados'] = df['equipos_afectados'].astype('int')
    df['coste'] = df['coste'] * 10000
    df['coste'] = df['coste'].astype('int')
    df.head(10)
    """
   Construir el modelo
   """
    lin_reg = LinearRegression()
    lin_reg.fit(df['equipos_afectados'].values.reshape((-1, 1)), df['coste'].values)
    lin_reg.intercept_
    lin_reg.coef_
    X_min_max = np.array([[df['equipos_afectados'].min()], [df['equipos_afectados'].max()]])
    y_train_pred = lin_reg.predict(X_min_max)
    """
   a predecir
   """
    x_new = np.array([[2500]])
    coste = lin_reg.predict(x_new)
    print(int(coste[0]), '$')
    """
   visualizar en grafico
   """
    plt.plot(df['equipos_afectados'], df['coste'], "b.")
    plt.plot(X_min_max, y_train_pred, "g-")
    plt.plot(x_new, coste, "rx")
    plt.xlabel("Numeros de equipos")
    plt.ylabel("Costo de reparacion")
    plt.show()
