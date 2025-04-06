from tensorflow.keras import layers, Input, Model

input_shape = (13, 200, 1)
num_classes = 5529
inputs = Input(shape=input_shape, name='input_layer')
#CNN
x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
x = layers.MaxPool2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPool2D((2,2))(x)
#Ajuste de dimensiones
x = layers.Reshape((-1,64))(x)
#RNN
x = layers.GRU(128, return_sequences=True)(x)
x = layers.GRU(128)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs, name='voice_model')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.save('voice_model.keras')