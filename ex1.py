import cv2
import os
import sqlite3
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Initialize the mixer for sound alerts
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades for face and eye detection
face = cv2.CascadeClassifier('haar cascade files\\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\\haarcascade_righteye_2splits.xml')

# Load the pre-trained model
model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Database setup
db_path = os.path.join(path, 'users.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)
""")
conn.commit()

class DrowsinessDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection")
        self.root.geometry("800x600")

        self.label = tk.Label(root)
        self.label.pack()

        self.start_button = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_button.pack()

        self.stop_button = tk.Button(root, text="Stop Detection", command=self.stop_detection)
        self.stop_button.pack()

        self.detection_running = False

        # Initialize variables
        self.score = 0
        self.thicc = 2
        self.rpred = [99]
        self.lpred = [99]

    def start_detection(self):
        self.detection_running = True
        self.detect_drowsiness()

    def stop_detection(self):
        self.detection_running = False

    def detect_drowsiness(self):
        if self.detection_running:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture video")
                return

            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
            left_eye = leye.detectMultiScale(gray)
            right_eye = reye.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

            for (x, y, w, h) in right_eye:
                r_eye = frame[y:y + h, x:x + w]
                r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye, (24, 24))
                r_eye = r_eye / 255
                r_eye = r_eye.reshape(24, 24, -1)
                r_eye = np.expand_dims(r_eye, axis=0)
                self.rpred = np.argmax(model.predict(r_eye), axis=-1)
                if self.rpred[0] == 1:
                    lbl = 'Open'
                if self.rpred[0] == 0:
                    lbl = 'Closed'
                break

            for (x, y, w, h) in left_eye:
                l_eye = frame[y:y + h, x:x + w]
                l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
                l_eye = cv2.resize(l_eye, (24, 24))
                l_eye = l_eye / 255
                l_eye = l_eye.reshape(24, 24, -1)
                l_eye = np.expand_dims(l_eye, axis=0)
                self.lpred = np.argmax(model.predict(l_eye), axis=-1)
                if self.lpred[0] == 1:
                    lbl = 'Open'
                if self.lpred[0] == 0:
                    lbl = 'Closed'
                break

            if self.rpred[0] == 0 and self.lpred[0] == 0:
                self.score += 1
                cv2.putText(frame, "Closed", (10, height - 20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                self.score -= 1
                cv2.putText(frame, "Open", (10, height - 20), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

            if self.score < 0:
                self.score = 0
            cv2.putText(frame, 'Score:' + str(self.score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            if self.score > 15:
                cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
                try:
                    sound.play()
                except:
                    pass
                self.thicc = self.thicc + 2
                if self.thicc > 16:
                    self.thicc = self.thicc - 2
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), self.thicc)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
            self.label.after(10, self.detect_drowsiness)

class LoginApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Login")
        self.root.geometry("300x200")
        
        self.label1 = tk.Label(root, text="Username")
        self.label1.pack()
        self.username_entry = tk.Entry(root)
        self.username_entry.pack()

        self.label2 = tk.Label(root, text="Password")
        self.label2.pack()
        self.password_entry = tk.Entry(root, show="*")
        self.password_entry.pack()

        self.login_button = tk.Button(root, text="Login", command=self.login)
        self.login_button.pack()
        
        self.signup_button = tk.Button(root, text="Signup", command=self.signup)
        self.signup_button.pack()

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        if user:
            messagebox.showinfo("Login", "Login successful!")
            self.root.destroy()
            self.open_detection_app()
        else:
            messagebox.showerror("Login", "Invalid username or password")

    def signup(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            messagebox.showinfo("Signup", "Signup successful!")
        except sqlite3.IntegrityError:
            messagebox.showerror("Signup", "Username already exists")

    def open_detection_app(self):
        root = tk.Tk()
        app = DrowsinessDetectionApp(root)
        root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = LoginApp(root)
    root.mainloop()
