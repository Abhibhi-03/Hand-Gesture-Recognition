import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model
model = load_model("final_gesture_action_model.keras")

# Define the class names
class_names = ["OK", "PALM_IN", "PALM_OUT", "THUMBS_UP", "THUMBS_DOWN"]

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Function to preprocess landmarks for prediction
def preprocess_landmarks(landmarks):
    # Flatten and normalize landmarks
    return np.array(landmarks).flatten()

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for mirror view
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw ROI box
    ROI_TOP_LEFT = (150, 100)
    ROI_BOTTOM_RIGHT = (500, 400)
    cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 0), 2)

    # Process frame for hand landmarks
    roi = frame[ROI_TOP_LEFT[1]:ROI_BOTTOM_RIGHT[1], ROI_TOP_LEFT[0]:ROI_BOTTOM_RIGHT[0]]
    results = hands.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    # Predict gesture if landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark data
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            input_data = preprocess_landmarks(landmarks)
            input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

            # Predict gesture
            predictions = model.predict(input_data)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions)

            # Display prediction
            cv2.putText(frame, f"Gesture: {class_names[predicted_class]}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Gesture Recognition", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
