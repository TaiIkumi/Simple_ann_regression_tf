import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_excel('Datos.xlsx', sheet_name='Hoja1')

# Selecciona los datos de entrada y salida del DataFrame
X = df["input"].values
y = df['output'].values

# Normaliza los datos de entrada
#X = (X - X.min()) / (X.max() - X.min())
#y = (y - y.min()) / (y.max() - y.min())

# Convierte los datos de entrada y salida a tensores
X = tf.constant(X)
y = tf.constant(y)

# Crea el modelo de la red neuronal
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(1,)))
model.add(tf.keras.layers.Dense(1, activation='linear'))

# Compila el modelo
model.compile(loss='mae', optimizer='Adam', metrics="mae")

# Entrena el modelo
model.fit(X, y, epochs=15000)

# Predice los valores
y_pred = model.predict(X)

# Grafica los datos de entrada y salida junto con la predicción de la red neuronal
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.show()

# Extraer pesos y bias
w, b = model.layers[0].get_weights()
print("w = ", w)
print("b = ", b)

# Guarda el modelo
model.save('model.h5')

# Carga el modelo
model = tf.keras.models.load_model('model.h5')




