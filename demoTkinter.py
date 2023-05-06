from tkinter import *
from PIL import Image, ImageTk
import os
import tkinter as tk
import cv2
import mediapipe as mp
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from firebase import firebase
import datetime as dt
from fpdf import FPDF
import firebase_admin
from firebase_admin import credentials, db
from firebase import firebase
import pandas as pd
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

window = tk.Tk()

window.title("FitPhysioPro")
window.geometry("1200x600")
window.resizable(False, False)
fontTuple = ("Nunito Bold", 24)
firebaset = ""
url = ""
urlRL = ""
uniqueDT = ""


def curls():
    import cv2
    import mediapipe as mp
    import numpy as np
    global firebaset, url, urlRL, uniqueDT
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    x = dt.datetime.now()
    uniqueDT = x.strftime("%H") + x.strftime("%M")
    urlRL = x.strftime("%H") + x.strftime("%M") + x.strftime("%S")
    url = '/Data/' + uniqueDT + "/"
    firebaset = firebase.FirebaseApplication('https://fitphysiopro-4fb5e-default-rtdb.firebaseio.com', None)

    # for lndmrk in mp_pose.PoseLandmark:
    #     print(lndmrk)

    def pushDataForReport(counter):
        global firebaset, url, urlRL, uniqueDT
        x = dt.datetime.now()
        data = {'time': x.strftime("%X"),
                'day': x.strftime("%A"),
                'date': dt.date.today(),
                'counter': counter
                }

        firebaset.post(url, data)

        # calculating Angle

    def calculate_angle(a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0
    stage = None

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                shoulder1 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow1 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist1 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                angle1 = calculate_angle(shoulder1, elbow1, wrist1)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                cv2.putText(image, str(angle1),
                            tuple(np.multiply(elbow1, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle and angle1 > 160:
                    stage = "down"
                    cv2.putText(image, "CORRECT POSTURE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                                              )
                if angle and angle1 < 30 and stage == 'down':
                    stage = "up"
                    counter += 1

                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                                              )
                    print(counter)
                    pushDataForReport(counter)


            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 80), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            # pushDataForReport(counter)

            cv2.rectangle(image, (250, 0), (360, 50), (0, 0, 0), -1)
            cv2.putText(image, 'CURLS', (250, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (100, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (70, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            cv2.imshow('FitPhysioPro', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def squats():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0
    stage = None

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                def calculate_angle(a, b, c):
                    a = np.array(a)  # First
                    b = np.array(b)  # Mid
                    c = np.array(c)  # End

                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians * 180.0 / np.pi)

                    if angle > 180.0:
                        angle = 360 - angle

                    return angle

                # Get coordinates
                kneeR = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                kneeL = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                hipR = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                hipL = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankleR = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                ankleL = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angle
                angle = calculate_angle(hipR, kneeR, ankleR)
                angle1 = calculate_angle(hipL, kneeL, ankleL)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(kneeR, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(angle1),
                            tuple(np.multiply(kneeL, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle < 60:
                    stage = "down"
                    cv2.putText(image, "CORRECT POSTURE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(colorr=(0, 255, 0), thickness=2, circle_radius=2)
                                              )

                if angle1 > 120 and stage == 'down':
                    stage = "up"
                    counter += 1
                    # pushDataForReport(counter)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(colorr=(0, 255, 0), thickness=2, circle_radius=2)
                                              )
                    print(counter)

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            cv2.imshow('FitPhysioPro', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def R_lunges():
    import cv2
    import mediapipe as mp
    import numpy as np
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                def calculate_angle(a, b, c):
                    a = np.array(a)  # First
                    b = np.array(b)  # Mid
                    c = np.array(c)  # End

                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians * 180.0 / np.pi)

                    if angle > 180.0:
                        angle = 360 - angle

                    return angle

                # Get coordinates
                kneeR = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                kneeL = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                hipR = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                hipL = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankleR = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                ankleL = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angle
                angle = calculate_angle(hipR, kneeR, ankleR)
                angle1 = calculate_angle(hipL, kneeL, ankleL)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(kneeR, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(angle1),
                            tuple(np.multiply(kneeL, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle < 130 and angle1 < 120:
                    stage = "down"
                    cv2.putText(image, "CORRECT POSTURE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if angle and angle > 150 and stage == "down":
                    stage = "up"
                    counter += 1

                    print(counter)
                    # pushDataForReport(counter)

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(image, (250, 0), (420, 50), (0, 0, 0), -1)
            cv2.putText(image, 'R-Lunges', (250, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (100, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (70, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                                      )

            cv2.imshow('FitPhysioPro', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def L_lunges():
    import cv2
    import mediapipe as mp
    import numpy as np
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                def calculate_angle(a, b, c):
                    a = np.array(a)  # First
                    b = np.array(b)  # Mid
                    c = np.array(c)  # End

                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians * 180.0 / np.pi)

                    if angle > 180.0:
                        angle = 360 - angle

                    return angle

                # Get coordinates
                kneeR = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                kneeL = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                hipR = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                hipL = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankleR = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                ankleL = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angle
                angle = calculate_angle(hipR, kneeR, ankleR)
                angle1 = calculate_angle(hipL, kneeL, ankleL)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(kneeR, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(angle1),
                            tuple(np.multiply(kneeL, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle < 120 and angle1 < 130:
                    stage = "down"
                    cv2.putText(image, "CORRECT POSTURE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if angle and angle > 150 and stage == "down":
                    stage = "up"
                    counter += 1

                    print(counter)
                    # pushDataForReport(counter)

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(image, (250, 0), (420, 50), (0, 0, 0), -1)
            cv2.putText(image, 'R-Lunges', (250, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (100, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (70, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                                      )

            cv2.imshow('FitPhysioPro', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def newWindow():
    root1 = Toplevel(window)
    root1.title("Posture Detection")
    root1.geometry("500x500")

    image = Image.open("curls.jpg")
    button_image = ImageTk.PhotoImage(image)

    button = Label(root1, image=button_image)
    button.place(x=30, y=70)

    image1 = Image.open("squats.jpg")
    button_image1 = ImageTk.PhotoImage(image1)

    button1 = Label(root1, image=button_image1)
    button1.place(x=30, y=160)

    image2 = Image.open("l_lunges.jpg")
    button_image2 = ImageTk.PhotoImage(image2)

    button2 = Label(root1, image=button_image2)
    button2.place(x=30, y=250)

    image3 = Image.open("r_lunges.jpg")
    button_image3 = ImageTk.PhotoImage(image3)

    button3 = Label(root1, image=button_image3)
    button3.place(x=30, y=340)

    text_label = tk.Label(root1, text="Exercises", width=9, height=1, font=("Nunito", 20))
    text_label.place(x=170, y=10)

    # create the button and associate the function with it
    button = tk.Button(root1, text="Curls", width=40, height=5, font=("Times", 10), command=curls)
    button.place(x=170, y=70)
    button1 = tk.Button(root1, text="Squats", width=40, height=5, font=("Times", 10), command=squats)
    button1.place(x=170, y=160)
    button1 = tk.Button(root1, text="Right Lunges", width=40, height=5, font=("Times", 10), command=R_lunges)
    button1.place(x=170, y=250)
    button1 = tk.Button(root1, text="Left Lunges", width=40, height=5, font=("Times", 10), command=L_lunges)
    button1.place(x=170, y=340)

    # start the main event loop
    root1.mainloop()


def generatePDF():
    cred = credentials.Certificate('fitphysiopro-4fb5e-firebase-adminsdk-cigbz-e8da8b3761.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://fitphysiopro-4fb5e-default-rtdb.firebaseio.com'
    })
    ref = db.reference('/Data')
    last_record = db.reference('/Data').order_by_key().limit_to_last(1).get()

    m = 10
    pw = 210 - 2 * m
    ch = 10
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 15)

    pdf.image("physiopro.jpg", x=20, y=10)

    pdf.cell(w=0, h=20, txt="FitPhysioPro Report", ln=1, align='C')

    pdf.cell(w=0, h=20, txt="Exercise : Dumbbells Curls", ln=1, align='C')

    pdf.cell(w=(pw / 4), h=ch, txt="Count", border=1, ln=0, align='C')
    pdf.cell(w=(pw / 4), h=ch, txt="Time", border=1, ln=0, align='C')
    pdf.cell(w=(pw / 4), h=ch, txt="Day", border=1, ln=0, align='C')
    pdf.cell(w=(pw / 4), h=ch, txt="Date", border=1, ln=1, align='C')

    totalCount = 0
    for user in last_record.values():
        for l in user.values():
            pdf.cell(w=(pw / 4), h=ch, txt=str(l["counter"]), border=1, ln=0)
            pdf.cell(w=(pw / 4), h=ch, txt=l["time"], border=1, ln=0)
            pdf.cell(w=(pw / 4), h=ch, txt=l["day"], border=1, ln=0)
            pdf.cell(w=(pw / 4), h=ch, txt=l["date"], border=1, ln=1)
            totalCount = l["counter"]

    pdf.cell(w=0, h=10, txt="Total Repetitions :" + str(totalCount), ln=20, align='C', border=True)

    l_record = list(last_record.items())[0][1]
    print(l_record)
    df = pd.DataFrame.from_dict(l_record, orient='index')
    df.drop(df.columns[[1, 2]], inplace=True, axis=1)

    print(df)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(df.time, df.counter)

    plt.title('Exercise Analysis')

    plt.xticks(rotation=30, ha='right')

    plt.xlabel('Time')
    plt.ylabel('Repetitions')
    fig.savefig("timeseries.png", dpi=100)

    pdf.image("./timeseries.png", x=0, y=130)

    pdf.output(f'./FitPhysioReport.pdf', 'F')
    os.system("FitPhysioReport.pdf")

    window.destroy()


bg = PhotoImage(file="background.png")
canvas1 = Canvas(window, width=1200, height=600)
canvas1.pack(fill="both")
canvas1.create_image(-1, -1, image=bg, anchor="nw")

helloTxt = Label(window, text="Hello there!", bg="white", font=fontTuple)
helloTxt.place(x=800, y=50)

l1 = Label(window, text="Welcome to FitPhysioPro,", bg="#00a2e8", fg="WHITE", font=("Nunito bold", 18))
l2 = Label(window, text="The ultimate AI assisted Physiotherapy training app.", font=("Nunito", 14), bg="#00a2e8",
           fg="WHITE")
l3 = Label(window,
           text="Whether you are recovering from an injury, seeking relief \nfrom chronic pain,or  simply looking  to  enhance your\nphysical performance our  app is here to  help you achieve\nyour goal.",
           justify=LEFT, font=("Nunito", 13), bg="#00a2e8", fg="WHITE")
l4 = Label(window,
           text="This includes wide range of exercises and  techniques\ndesigned and recommendations by  Certified Physiotherapists\nto target  specific muscle  groups .Exercises can be customized\nto meet your unique needs.  Detailed exercise  reports are \ngenerated to help you assess  your  accuracy  and progress\nwhile  exercising.Reports can be shared with Specialized\nPhysiotherapists for   monitoring progress and feedbacks. ",
           justify=LEFT, font=("Nunito", 13), bg="#00a2e8", fg="WHITE")
l5 = Label(window,
           text="This muscle mentor is all you need to recover your muscle\nailments, injuries,increase flexibility and strength building...!!!",
           justify=LEFT, font=("Nunito", 13), bg="#00a2e8", fg="WHITE")

l1.place(x=140, y=60)
l2.place(x=50, y=120)
l3.place(x=50, y=170)
l4.place(x=50, y=280)
l5.place(x=50, y=450)

imgExercise = ImageTk.PhotoImage(Image.open("5.png"))

logo = Label(window, image=imgExercise, bg="white")
logo.place(x=790, y=90)

startExerciseBtn = tk.Button(window, text="Start Exercising", font=("Nunito", 16), command=newWindow)
startExerciseBtn.place(x=800, y=260)

imgReport = ImageTk.PhotoImage(Image.open("6.png"))
logo = Label(window, image=imgReport, bg="white")
logo.place(x=790, y=320)
generateBtn = tk.Button(window, text="Generate Report", font=("Nunito", 16), command=generatePDF)
generateBtn.place(x=800, y=500)

window.mainloop()
