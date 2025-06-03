import cv2
import os
import time
from datetime import datetime
from picamera2 import Picamera2

# Verander dit naar de emotie die je wilt vastleggen
EMOTION_NAME = "happy"  

def create_emotion_folder(emotion):
    """Maakt een dataset-map en een submap voor de specifieke emotie aan."""
    dataset_folder = "/home/injo/Emotion_Recognition/dataset"
    
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)  # Hoofdmap voor dataset
    
    emotion_folder = os.path.join(dataset_folder, emotion)
    if not os.path.exists(emotion_folder):
        os.makedirs(emotion_folder)  # Submap voor de emotie
    
    return emotion_folder

def capture_photos(emotion):
    """Start de camera en slaat foto's op in de juiste map."""
    folder = create_emotion_folder(emotion)
    
    # Initialiseer de Raspberry Pi Camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()
    
    time.sleep(2)  # Geef de camera tijd om op te starten
    photo_count = 0
    
    print(f"Foto's maken voor {emotion}. Druk op SPATIE om een foto te nemen, 'q' om te stoppen.")

    while True:
        frame = picam2.capture_array()  # Neem een frame van de camera
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converteer naar grijswaarden
        
        # Toon de live camera-feed
        cv2.imshow('Capture', gray)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Spatiebalk om een foto te nemen
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{emotion}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, gray)  # Opslaan als grijswaardenafbeelding
            print(f"Foto {photo_count} opgeslagen: {filepath}")

        elif key == ord('q'):  # 'q' om te stoppen
            break

    # Cleanup
    cv2.destroyAllWindows()
    picam2.stop()
    print(f"Foto-opname voltooid. {photo_count} foto's opgeslagen voor {emotion}.")

if __name__ == "__main__":
    capture_photos(EMOTION_NAME)
