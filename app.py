import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import pygame
import cv2
import numpy
import mediapipe as mp
from scipy.spatial import distance as dist


LEFT_EYE_INDEXES = [362, 386, 387, 263, 373, 374]
RIGHT_EYE_INDEXES = [33, 159, 158, 133, 145, 144]


def eye_aspect_ratio(P):
    try:
        A = dist.euclidean(P[1], P[5])
        B = dist.euclidean(P[2], P[4])
        C = dist.euclidean(P[0], P[3])

        ear = (A + B) / (2.0 * C)
        return ear
    
    except:
        return -1


class DrowsyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("700x650")
        self.title("Driver Monitoring Systems")
        self.create_widgets()

    def create_widgets(self):
        # Khung video
        vid_frame = tk.Frame(self, height=480, width=600)
        vid_frame.pack()
        self.vid_label = tk.Label(vid_frame)
        self.vid_label.pack()

        # display information about YOLOv8 processes each frame 
        # self.info_label_yolo = tk.Label(self, text="", font=("Helvetica", 12))
        # self.info_label_yolo.pack()
        
        # load YOLO model
        self.threshold = 0.5
        self.model = YOLO("weights/best.pt")

        # load mediapipe model
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=1)

        # Initialize MediaPipe Drawing Utilities.
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=[125, 125, 125])

        # Bắt đầu luồng video
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()

        self.counter = 0
        self.threshold_EAR = 27
        self.blinkhalf_closed = 0
        self.eye_open = 0

        pygame.init()
        pygame.mixer.init()        
        self.sound_file = "warning.wav"
        self.sound = pygame.mixer.Sound(self.sound_file)

    def play_warning_sound(self):
        self.sound.play()

    def update_video(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, original_frame = cap.read()
            original_frame = cv2.flip(original_frame, 1)
            height, width, channels = original_frame.shape

            # frame is predicted by YOLO and MediaPipe
            results = self.model.predict(original_frame, verbose=False, max_det = 1, conf = self.threshold, line_width=2) 
            results_mp = self.face_mesh.process(original_frame)

            display_frame = results[0].plot().copy()

            if ret:    
                # YOLO warning   
                if results[0].boxes.cpu().numpy().cls.size > 0:
                    if numpy.any(results[0].boxes.cpu().numpy().cls == 0):
                        print('counter:', self.counter)
                        if self.counter < 20:
                            self.counter += 1
                        # maximun counter is 19, if counter is 20 , play warning sound
                        if not pygame.mixer.get_busy() and self.counter >= 20:
                            self.play_warning_sound()
                    elif self.blinkhalf_closed < 25:
                        self.counter = 0
                        pygame.mixer.stop() 
                else:
                    if not pygame.mixer.get_busy():
                            self.play_warning_sound()

                if results_mp.multi_face_landmarks:
                    p_left = []
                    p_right = []

                    face_landmarks = results_mp.multi_face_landmarks[0]

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

                    # for face_landmarks in results_mp.multi_face_landmarks:
                    #     self.mp_drawing.draw_landmarks(
                    #         image=display_frame,
                    #         landmark_list=face_landmarks,
                    #         connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    #         landmark_drawing_spec=self.drawing_spec,
                    #         connection_drawing_spec=self.drawing_spec)
                        
                
                    ear_left = eye_aspect_ratio(p_left)
                    ear_right = eye_aspect_ratio(p_right)
                    ratio = 0.5*(ear_left + ear_right)*100


                    if self.eye_open >= 15:
                        pygame.mixer.stop() 
                        self.blinkhalf_closed = 0
                        self.eye_open = 0

                    if ratio <= self.threshold_EAR:
                        if self.blinkhalf_closed < 25:
                            self.blinkhalf_closed += 1
                            print('+eye+', self.blinkhalf_closed)
                        if self.blinkhalf_closed >= 25:
                            if not pygame.mixer.get_busy():
                                self.play_warning_sound()
                    elif self.blinkhalf_closed >= 25:
                        if self.eye_open < 15:
                            self.eye_open += 1
                        
                # display YOLO frame onto interface
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(Image.fromarray(display_frame))
                self.vid_label.config(image=photo)
                self.vid_label.image = photo
                

            # # display information about YOLOv8 processes each frame on self.info_label_yolo
            # if results is not None and len(results) > 0:
            #     speed_info = results[0].speed
            #     info_text = "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms postprocess per image at shape {}".format(
            #         speed_info["preprocess"],
            #         speed_info["inference"],
            #         speed_info["postprocess"],
            #         results[0].orig_shape
            #     )
            #     self.info_label_yolo.config(text=info_text)
            

            self.update()

        # Ngừng quay video khi đóng ứng dụng
        cap.release()


# Chạy ứng dụng
if __name__ == "__main__":
    app = DrowsyApp()
    app.mainloop()