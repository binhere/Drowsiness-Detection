import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
from cvzone.PlotModule import LivePlot
from ultralytics import YOLO


def eye_aspect_ratio(P):
    A = dist.euclidean(P[1], P[5])
    B = dist.euclidean(P[2], P[4])
    C = dist.euclidean(P[0], P[3])

    ear = (A + B) / (2.0 * C)
    return ear


model = YOLO("weights/best.pt")

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=1, static_image_mode=False)

# Initialize MediaPipe Drawing Utilities.
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=[125, 125, 125])

# Open the camera.
cap = cv2.VideoCapture(0)

plotY = LivePlot(640, 480, [0, 40])

MINIMUM_EAR = 27
LEFT_EYE_INDEXES = [362, 386, 387, 263, 373, 374]
RIGHT_EYE_INDEXES = [33, 159, 158, 133, 145, 144]

font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 0.4
color = (255, 0, 0) 
thickness = 1

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    height, width, channels = frame.shape

    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_mp = face_mesh.process(image)
    results = model.predict(frame, verbose=False, max_det = 1, conf = 0.3, line_width=2)

    display_frame = results[0].plot().copy()

    # Draw the facial landmarks on the image.
    if results_mp.multi_face_landmarks:
        # for face_landmarks in results_mp.multi_face_landmarks:
        #     mp_drawing.draw_landmarks(
        #         image=frame_mp,
        #         landmark_list=face_landmarks,
        #         connections=mp_face_mesh.FACEMESH_TESSELATION,
        #         landmark_drawing_spec=drawing_spec,
        #         connection_drawing_spec=drawing_spec)
        
        face_landmarks = results_mp.multi_face_landmarks[0]
        
        p_left = []
        p_right = []

        ear_left = 0
        ear_right = 0

        for id_ldmrk in LEFT_EYE_INDEXES:
                pos = face_landmarks.landmark[id_ldmrk]  
                x = int(pos.x * width)
                y = int(pos.y * height) 
                p_left.append((x, y))
                cv2.circle(display_frame, (x, y), 1, (0, 255, 0), 1)

        for id_ldmrk in RIGHT_EYE_INDEXES:
                pos = face_landmarks.landmark[id_ldmrk]  
                x = int(pos.x * width)
                y = int(pos.y * height)
                p_right.append((x, y))
                cv2.circle(display_frame, (x, y), 1, (0, 255, 0), 1)
                
        ear_left = eye_aspect_ratio(p_left)
        ear_right = eye_aspect_ratio(p_right)
        ratio = 0.5*(ear_left + ear_right)*100

        if ratio <= MINIMUM_EAR:
           print(ratio)

        cv2.imshow('plotY', plotY.update(ratio))

    # Display the image
    cv2.imshow('YOLOv8 - Mediapile', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()