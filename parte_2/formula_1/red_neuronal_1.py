import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# a) Construcción del conjunto de datos y archivo CSV
num_samples = 1000
x1 = np.random.rand(num_samples)
x2 = np.random.rand(num_samples)
x3 = np.random.rand(num_samples)
a = 0.1
b = 2.3
c = 0.02
y = a * x1 + b * (x2 ** 2) - x3 * c
data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'y': y})
data.to_csv('datos.csv', index=False)

# b) Diseño de la red neuronal
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(3,)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# c) División en conjuntos de entrenamiento y prueba
data = pd.read_csv('datos.csv')
x = data[['x1', 'x2', 'x3']].values
y = data['y'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# d) Entrenamiento de la red neuronal con gráfico de avance
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# e) Grabado de la red neuronal entrenada en archivos
model.save_weights('model_weights.h5')
model.save('model_architecture.h5')

# f) Apertura de una red neuronal desde un archivo
model = load_model('model_architecture.h5')
model.load_weights('model_weights.h5')

# g) Utilización y gráfico con PyPlot
y_pred = model.predict(x_test)
plt.scatter(range(len(y_test)), y_test, label='Actual')
plt.scatter(range(len(y_pred)), y_pred, label='Predicted')
plt.xlabel('Sample')
plt.ylabel('y')
plt.legend()
plt.show()