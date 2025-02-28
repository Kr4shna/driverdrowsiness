import cv2
import mediapipe as mp
import numpy as np

# Setup MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from IP camera
cap = cv2.VideoCapture("http://192.168.187.182:8080///video")  # Replace with your IP address

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the EAR threshold for drowsiness
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20

# Initialize the counter for drowsiness
counter = 0

# Helper function to calculate EAR
def calculate_ear(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture video.")
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmarks for eyes
            h, w, _ = frame.shape
            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in [362, 385, 387, 263, 373, 380]]
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in [33, 160, 158, 133, 153, 144]]

            # Draw landmarks on eyes
            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Calculate EAR for both eyes
            left_ear = calculate_ear(np.array(left_eye))
            right_ear = calculate_ear(np.array(right_eye))
            ear = (left_ear + right_ear) / 2.0

            # Check if the EAR is below the threshold
            if ear < EAR_THRESHOLD:
                counter += 1
                if counter >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "Drowsiness Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                counter = 0

            # Display EAR on frame
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Drowsiness Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
