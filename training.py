import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

with open('mfcc.pkl', 'rb') as f:
    data = pickle.load(f)

print(type(data))  # Debería ser una lista
print(len(data))   # Cantidad de elementos
print(type(data[0]))  # Cada entrada debería ser un diccionario
print(data[0])  # Inspecciona la primera entrada

X = [entry["mfcc"] for entry in data]
y = np.array(range(len(X)))
X = np.array(X)

num_classes = len(set(y))
y = to_categorical(y, num_classes=num_classes)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
model = tf.keras.models.load_model('voice_model.keras')

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)
model.save('trained_voice_model.keras')