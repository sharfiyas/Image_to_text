import cv2
import numpy as np
from keras.models import model_from_json
import tensorflow as tf

# Dictionary to label all emotions
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the model
emotion_model = tf.keras.models.load_model('emotion_model.h5')

while True:
    cap = None
    for index in range(4):  # Try first 4 camera indexes
        cap = cv2.VideoCapture(index)
        ret, frame = cap.read()
        if ret:
            print(f"Camera found at index {index}")
            break
        cap.release()

    if not ret:
        print("No camera found.")
        break

    # Resize the frame if successfully captured
    frame = cv2.resize(frame, (1080, 720))
    
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load OpenCV face detector
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Detect faces
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display frame with emotion labels
    cv2.imshow('Emotion Detection', frame)
    
    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
