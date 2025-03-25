import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
from tkinter import Label, Button, StringVar
from PIL import Image, ImageTk
import pyttsx3  


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Diccionario de las señas
labels_dict = {
    0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U', 5: 'HOLA', 6: 'H', 7: 'L', 8: 'M', 
    9: 'BIEN', 10: 'MAL', 11: 'PERMISO', 12: 'TU', 13: '4', 14: 'PADRE', 15: 'MADRE', 
    16: 'ESTUDIANTE', 17: 'PELO', 18: 'MANO', 19: 'EL SALVADOR'
}


sentence = []


last_prediction_time = time.time()  
debounce_time = 3 


first_prediction_done = False


root = tk.Tk()
root.title("Reconocimiento de Señal en Lengua de Señas")


sentence_var = StringVar()
sentence_var.set('Frase: ')


label_sentence = Label(root, textvariable=sentence_var, font=("Helvetica", 16), width=50, height=2)
label_sentence.pack()

# Crear un botón para resetear la frase
def reset_sentence():
    global sentence
    sentence = []
    sentence_var.set('Frase: ')
    print("Frase reseteada")

button_reset = Button(root, text="Resetear Frase", font=("Helvetica", 14), command=reset_sentence)
button_reset.pack()

# convertir a voz
engine = pyttsx3.init()


engine.setProperty('rate', 150)  # 150 palabras por minuto en este caso


def update_frame(frame):
   
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    label_video.config(image=img_tk)
    label_video.image = img_tk

    root.after(10, show_frame)


label_video = Label(root)
label_video.pack()

# Función para detectar y mostrar la predicción de la seña
def show_frame():
    global sentence, last_prediction_time, first_prediction_done

    ret, frame = cap.read()

    if not ret:
        print("No se pudo obtener el frame de la cámara")
        return

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,  # puntos de la mano que se detectan
                mp_hands.HAND_CONNECTIONS,  
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # Obtener las 42 características
            x_ = []
            y_ = []
            data_aux = []
            for i in range(21):  # Hay 21 puntos en la mano
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

                
                data_aux.append(x - min(x_))  
                data_aux.append(y - min(y_)) 

            if len(data_aux) == 42:
                
                prediction = model.predict([np.asarray(data_aux)])# Realizar la predicción con el modelo

                predicted_character = labels_dict.get(int(prediction[0]), 'Desconocido')

                # Controlar el tiempo entre las predicciones
                current_time = time.time()
                if current_time - last_prediction_time > debounce_time:
                   
                    sentence.append(predicted_character) # Si la predicción es nueva, la agregamos a la lista de frases
                    last_prediction_time = current_time

                    
                    if first_prediction_done:
                       
                        engine.say(predicted_character)
                        engine.runAndWait() # Leer en voz alta la predicción 

                    # Después de la primera predicción, activamos el flag
                    else:
                        first_prediction_done = True

                # Mostrar la predicción en la pantalla
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Actualizar la frase en la interfaz gráfica
                sentence_var.set('Frase: ' + ' '.join(sentence))

    update_frame(frame)


root.after(10, show_frame)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
