import os
import cv2
import time
import vosk
import json
import pickle
import pyttsx3
import pyaudio
import threading
import numpy as np
import tkinter as tk
import mediapipe as mp

from PIL import Image, ImageTk
from tkinter import Label, Button, StringVar

sentence = []
debounce_time = 3
camera_on = True
listening = False
thread = None
first_prediction_done = False
last_prediction_time = time.time()

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
model_vosk = vosk.Model("vosk-model-small-es-0.42")
recognizer = vosk.KaldiRecognizer(model_vosk, 16000)
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

signal_dict = {
    0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U', 5: 'HOLA', 6: 'H', 7: 'L', 8: 'M', 
    9: 'BIEN', 10: 'MAL', 11: 'PERMISO', 12: 'TU', 13: '4', 14: 'PADRE', 15: 'MADRE', 
    16: 'ESTUDIANTE', 17: 'PELO', 18: 'MANO', 19: 'EL SALVADOR'
}

signal_path = 'señas/'
signals_images = {'hola':'hola.png', 'bien':'bien.png', 'mal':'mal.png',
                  'padre':'padre.png','madre':'madre.png', 'estudiante':'estudiante.png','pelo':'pelo.png',
                  'mano':'mano.png', 'el salvador':'el_salvador.png'
}

root = tk.Tk()
root.title("Trducción LESSA")

sentence_var = StringVar()
sentence_var.set('Frase: ')

label_sentence = Label(root, textvariable=sentence_var, font=("Helvetica", 16), width=50, height=2)
label_sentence.pack()

def reset_sentence():
    global sentence
    sentence = []
    sentence_var.set('Frase: ')

button_reset = Button(root, text="Resetear Frase", font=("Helvetica", 14), command=reset_sentence)
button_reset.pack()

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

def show_frame():
    global sentence, last_prediction_time, first_prediction_done

    if not camera_on: return

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
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,  
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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
                predicted_character = signal_dict.get(int(prediction[0]), 'Desconocido')

                current_time = time.time()

                if current_time - last_prediction_time > debounce_time:
                    sentence.append(predicted_character) # Si la predicción es nueva, la agregamos a la lista de frases
                    last_prediction_time = current_time

                    if first_prediction_done:
                        engine.say(predicted_character)
                        engine.runAndWait() # Leer en voz alta la predicción 

                    else:
                        first_prediction_done = True

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Actualiza la frase visible
                sentence_var.set('Frase: ' + ' '.join(sentence))

    update_frame(frame)

def stop_camera():
    global camera_on
    camera_on = False
    cap.release()
    cv2.destroyAllWindows()
    label_video.config(image="")

def start_camera():
    global camera_on, cap, listening
    camera_on = True
    stop_recognition()
    cap = cv2.VideoCapture(0)
    show_frame()

def show_signal(sentence):
        for word in sentence:
            print(word)
            image_name = signals_images.get(word.lower())
            image_path = os.path.join(signal_path, image_name)
            if image_path:
                img = Image.open(image_path)
                img = img.resize((300, 300))
                img_tk = ImageTk.PhotoImage(img)
                label_video.config(image=img_tk)
                label_video.image = img_tk
                root.update()
                time.sleep(0.5)

def recognize_speech():
    global listening
    listening = True
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
    stream.start_stream()
    audio_data = b""
    sentence_var.set("Escuchando...")
    sentence.clear()

    while listening:
        try:
            for _ in range(20):  # Captura ~3 segundos
                data = stream.read(4000, exception_on_overflow=False)
                audio_data += data
            if recognizer.AcceptWaveform(audio_data):
                result = json.loads(recognizer.Result())
                transcribed_text = result.get("text", "").strip()

                if transcribed_text:
                    sentence.append(transcribed_text)
                    sentence_var.set("Frase: " + " ".join(sentence))
                    show_signal(sentence)
                    sentence.clear()
                audio_data = b""
        except Exception as e:
            sentence_var.set("No fue posible capturar el audio.")
            print('Error de reconocimiento de voz',e)
            sentence.clear()
            continue
    stream.stop_stream()
    stream.close()
    p.terminate()

def start_recognition():
    global thread
    if camera_on: stop_camera()
    if not listening:
        thread = threading.Thread(target=recognize_speech, daemon=True)
        thread.start()

def stop_recognition():
    global listening, thread

    if listening:
        listening = False
        if thread:
            thread.join()  # Esperar a que el hilo termine
            thread = None

btn_voice_signal = Button(root, text="Traducir Voz", font=("Helvetica", 14), command=start_recognition)
btn_voice_signal.pack()

btn_voice_signal = Button(root, text="Traducir Seña", font=("Helvetica", 14), command=start_camera)
btn_voice_signal.pack()

root.after(10, show_frame)
root.mainloop()

cap.release()
cv2.destroyAllWindows()