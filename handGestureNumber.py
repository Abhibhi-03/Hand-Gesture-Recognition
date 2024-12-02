import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Initialize a deque to smooth landmarks
landmark_history = deque(maxlen=5)

# Define ROI box
ROI_TOP_LEFT = (150, 100)
ROI_BOTTOM_RIGHT = (500, 400)

# Smooth landmark function
def smooth_landmarks(current_landmarks):
    landmark_history.append(current_landmarks)
    return np.mean(np.array(landmark_history), axis=0)

# Logic to count raised fingers (for the number-hand)
def fingers_up(landmarks, handedness):
    fingers = []

    # Thumb
    if handedness == "Right":
        fingers.append(1 if landmarks[4][0] > landmarks[3][0] else 0)
    else:
        fingers.append(1 if landmarks[4][0] < landmarks[3][0] else 0)

    # Other fingers
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(1 if landmarks[tip][1] < landmarks[pip][1] else 0)

    return sum(fingers)

# Gesture logic (for the gesture-hand)
def identify_gesture(landmarks):
    # Example gesture logic based on landmark positions
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    pinky_tip = landmarks[20]

    # Simple logic to detect a "peace" sign (index and middle finger up, others down)
    if landmarks[8][1] < landmarks[6][1] and landmarks[12][1] < landmarks[10][1]:
        if landmarks[4][1] > landmarks[3][1] and landmarks[20][1] > landmarks[18][1]:
            return "Peace Sign"
    # Add other gestures as needed
    return "Unknown Gesture"

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 0), 2)
    cv2.putText(frame, "Place hand inside box", (ROI_TOP_LEFT[0], ROI_TOP_LEFT[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    hand_detected = False
    if results.multi_handedness and results.multi_hand_landmarks:
        for hand_handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
            handedness = hand_handedness.classification[0].label

            # Check if the hand is within the ROI
            x, y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
            abs_x, abs_y = int(x * frame.shape[1]), int(y * frame.shape[0])
            if not (ROI_TOP_LEFT[0] < abs_x < ROI_BOTTOM_RIGHT[0] and ROI_TOP_LEFT[1] < abs_y < ROI_BOTTOM_RIGHT[1]):
                continue

            hand_detected = True
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            smoothed_landmarks = smooth_landmarks(landmarks)

            if handedness == "Right":
                # Use the right hand for numbers
                fingers = fingers_up(smoothed_landmarks, handedness)
                cv2.putText(frame, f"Right Hand: {fingers} Fingers", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Use the left hand for gestures
                gesture = identify_gesture(smoothed_landmarks)
                cv2.putText(frame, f"Left Hand Gesture: {gesture}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if not hand_detected:
        cv2.putText(frame, "Place hand inside ROI!", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
