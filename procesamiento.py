import pickle

data_dict = pickle.load(open('./data.pickle', 'rb'))
# Recorta las listas con longitud 84 y ajusta las que tienen menos de 42
# 21 puntos de referencia * 2 coordenadas (x, y)
expected_length = 42

for i in range(len(data_dict['data'])):
    item = data_dict['data'][i]

    if len(item) == 84:
        data_dict['data'][i] = item[:expected_length]
        print(f"Elemento {i} recortado a 42: {data_dict['data'][i]}")
    elif len(item) < expected_length:
        data_dict['data'][i] = item + [0] * (expected_length - len(item))
        print(f"Elemento {i} completado con ceros: {data_dict['data'][i]}")

# Verificación después del ajuste
for i, item in enumerate(data_dict['data']):
    print(f"Longitud del elemento {i}: {len(item)}")

with open('./data_modified.pickle', 'wb') as f:
    pickle.dump(data_dict, f)