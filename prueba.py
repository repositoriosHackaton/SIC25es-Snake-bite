# Verifica la longitud de cada sublista
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))
# Recorta las listas con longitud 84 y ajusta las que tienen menos de 42
expected_length = 42  

for i in range(len(data_dict['data'])):
    item = data_dict['data'][i]
    
    if len(item) == 84:
        data_dict['data'][i] = item[:expected_length]
        print(f"Elemento {i} recortado a 42: {data_dict['data'][i]}")
    elif len(item) < expected_length:
        # añade ceros para igualarlas
        data_dict['data'][i] = item + [0] * (expected_length - len(item))
        print(f"Elemento {i} completado con 0: {data_dict['data'][i]}")

for i, item in enumerate(data_dict['data']):
    print(f"Longitud del elemento {i}: {len(item)}")


with open('./data_modified.pickle', 'wb') as f:
    pickle.dump(data_dict, f)
#este es el nuevo pickle que se guarda pq lo modifique porque me daba longitudes de 84 
print("Datos modificados se han guardado en 'data.pickle'.")
