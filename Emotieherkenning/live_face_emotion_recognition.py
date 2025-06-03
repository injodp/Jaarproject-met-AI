import cv2
import numpy as np
import tensorflow as tf
import face_recognition
import pickle
from picamera2 import Picamera2
import os
from datetime import datetime

# 🟢 Laad het gezichtsherkenningsmodel (pickle)
print("[INFO] Laden van gezichtsherkenning encodings...")
with open("/home/injo/Face Recognition/encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())

known_face_encodings = data["encodings"]
known_face_names = data["names"]
authorized_names = []
# 🟢 Laad het emotieherkenningsmodel (TensorFlow)
print("[INFO] Laden van emotieherkenningsmodel...")
emotion_model = tf.keras.models.load_model("/home/injo/Emotion_Recognition/emotion_recognition_custom_model.h5")
emotion_labels = ['happy', 'sad', 'angry', 'disgust', 'fear', 'surprise','neutral']  # Pas aan volgens jouw dataset

# 🟢 Initialiseer de camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cv_scaler = 4  # Schaal de video in om snelheid te verbeteren
face_locations = []
face_encodings = []
face_names = []
frame_count = 0

print("[INFO] Live detectie gestart... Druk op 'q' om te stoppen.")

while True:
    # 🟢 Neem een frame van de camera
    frame = picam2.capture_array()
    
    # 🟢 Verklein het frame voor snellere verwerking
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # 🟢 Gezichtsherkenning uitvoeren
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')

    face_names = []
    face_emotions = []

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # 🟢 Identificeer de persoon
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            if name not in authorized_names:
                inbreker_dir = "/home/injo/inbreker"
                if not os.path.exists(inbreker_dir):
                    os.mkdir(inbreker_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                inbreker = picam2.capture_file(f"{inbreker_dir}/inbreker-{timestamp}.jpg")
                print("picture taken")
        face_names.append(name)

        # 🟢 Emotie detecteren
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        face_roi = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = face_roi / 255.0  # Normaliseren

        predictions = emotion_model.predict(face_roi)
        emotion_index = np.argmax(predictions)
        emotion = emotion_labels[emotion_index]
        face_emotions.append(emotion)

    # 🟢 Resultaten tekenen op het scherm
    for (top, right, bottom, left), name, emotion in zip(face_locations, face_names, face_emotions):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        # 🟢 Gezichtsherkenningsbox tekenen
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        cv2.rectangle(frame, (left - 3, top - 70), (right + 3, top), (244, 42, 3), cv2.FILLED)
        
        log_file = "emotie_log.text"
        def log_emoties(name, emotion):
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M:%S")
        with open("/home/injo/Emotion_Recognition/emotie_log.text", "a") as f:
            f.write(f"{timestamp} - {name}: {emotion}\n")
            
        # 🟢 Naam en emotie op het scherm zetten
        label = f"{name} | {emotion}"
        cv2.putText(frame, label, (left + 6, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    # 🟢 Toon het beeld in een venster
    cv2.imshow("Gezichts- en Emotieherkenning", frame)

    # 🟢 Stop als de 'q' toets wordt ingedrukt
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 🟢 Ruim op na afsluiten
cv2.destroyAllWindows()
picam2.stop()
print("[INFO] Live detectie gestopt.")
