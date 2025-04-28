import cv2
import mediapipe as mp
import numpy as np
from IPython.display import display, clear_output
import ipywidgets as widgets

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
reps = 0
sets = 0
is_counting = False
last_gesture = None

# Function to detect thumbs up
def is_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    index_tip = hand_landmarks.landmark[8]
    return (thumb_tip.y < thumb_ip.y and thumb_tip.y < index_tip.y)

# Function to detect peace sign
def is_peace_sign(hand_landmarks):
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    thumb_tip = hand_landmarks.landmark[4]
    return (index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and
            ring_tip.y > middle_tip.y and pinky_tip.y > middle_tip.y)

# Function to process frame and detect gestures
def process_frame(frame):
    global reps, sets, is_counting, last_gesture
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Initialize gesture flags
    thumbs_up_count = 0
    peace_sign_count = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Detect gestures
            if is_thumbs_up(hand_landmarks):
                thumbs_up_count += 1
            elif is_peace_sign(hand_landmarks):
                peace_sign_count += 1
    
    # Gesture logic
    current_gesture = None
    if thumbs_up_count == 2:
        current_gesture = "thumbs_up"
        if not is_counting and last_gesture != "thumbs_up":
            is_counting = True
    elif peace_sign_count == 2:
        current_gesture = "peace_sign"
        if is_counting and last_gesture != "peace_sign":
            is_counting = False
            sets += 1
            reps = 0
    
    last_gesture = current_gesture
    
    # Simulate rep counting (replace with actual exercise detection logic)
    if is_counting:
        # Example: Increment reps every 30 frames (adjust based on exercise)
        frame_count = cv2.getTickCount() % 30
        if frame_count == 0:
            reps += 1
    
    # Display counters
    cv2.putText(frame, f"Sets: {sets}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Reps: {reps}", (frame.shape[1]-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create output widget for video display
output_widget = widgets.Image()
display(output_widget)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        frame = process_frame(frame)
        
        # Convert frame to JPEG for display
        _, buffer = cv2.imencode('.jpg', frame)
        output_widget.value = buffer.tobytes()
        
        # Clear output to update frame
        clear_output(wait=True)
        display(output_widget)
        
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    cap.release()
    hands.close()