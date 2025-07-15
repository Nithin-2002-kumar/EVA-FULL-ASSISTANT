# eva.py - EVA: Enhanced Voice Assistant (Full Integration)

import os
import sys
import cv2
import time
import faiss
import fitz  # PyMuPDF
import webbrowser
import subprocess
import threading
import logging
import pyttsx3
import pyautogui
import numpy as np
import speech_recognition as sr
from googletrans import Translator
from fer import FER
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from sentence_transformers import SentenceTransformer
import mediapipe as mp
from sentence_transformers.util import cos_sim
import spacy
import wikipedia

# Logging setup
logging.basicConfig(filename="assistant.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Text-to-speech setup
engine = pyttsx3.init('sapi5')
engine.setProperty("rate", 150)
engine.setProperty("volume", 0.9)

# Global variables
user_preferences = {'name': 'User', 'preferred_language': 'en'}
language_code = 'en'
dashboard = None
translator = Translator()
kb_index = None
kb_chunks = []
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# GUI Dashboard
class EvaDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EVA - Enhanced Voice Assistant")
        self.root.geometry("600x400")
        self.root.configure(bg="#1e1e1e")

        self.status_var = tk.StringVar(value="Idle...")
        self.last_command_var = tk.StringVar(value="Waiting for command...")

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="EVA Dashboard", font=("Helvetica", 20, "bold"), fg="#00ffcc", bg="#1e1e1e").pack(pady=10)
        tk.Label(self.root, text="Status:", font=("Arial", 12), bg="#1e1e1e", fg="white").pack()
        tk.Label(self.root, textvariable=self.status_var, font=("Arial", 12), fg="yellow", bg="#1e1e1e").pack()
        tk.Label(self.root, text="Last Command:", font=("Arial", 12), bg="#1e1e1e", fg="white").pack(pady=(20, 0))
        tk.Label(self.root, textvariable=self.last_command_var, font=("Arial", 12), fg="lightgreen", bg="#1e1e1e").pack()
        button_frame = tk.Frame(self.root, bg="#1e1e1e")
        button_frame.pack(pady=30)
        ttk.Button(button_frame, text="Mute", command=self.mute).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Exit", command=self.exit_eva).pack(side=tk.LEFT, padx=10)

    def update_status(self, status):
        self.status_var.set(status)

    def update_command(self, command):
        self.last_command_var.set(command)

    def mute(self):
        speak("Voice output muted.")
        engine.setProperty("volume", 0.0)

    def exit_eva(self):
        speak("Goodbye. Shutting down.")
        self.root.quit()
        os._exit(0)

    def run(self):
        self.root.mainloop()

# Speak function

def speak(text):
    try:
        translated = translator.translate(text, src='en', dest=language_code).text
        engine.say(translated)
        print(f"EVA (translated): {translated}")
        if dashboard:
            dashboard.update_status("Speaking...")
        engine.runAndWait()
    except Exception as e:
        print("Error:", str(e))

# Listen function

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
        except sr.WaitTimeoutError:
            return None
    try:
        text = recognizer.recognize_google(audio, language=language_code)
        translated = translator.translate(text, src=language_code, dest='en').text
        print(f"Translated: {translated}")
        return translated.lower()
    except (sr.UnknownValueError, sr.RequestError):
        return None

# Knowledge base loader

def load_knowledge_file(file_path):
    global kb_index, kb_chunks
    kb_chunks = []
    text = ""
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    sentences = text.split(". ")
    for i in range(0, len(sentences), 5):
        kb_chunks.append(". ".join(sentences[i:i+5]))

    embeddings = embedding_model.encode(kb_chunks)
    kb_index = faiss.IndexFlatL2(embeddings[0].shape[0])
    kb_index.add(embeddings)
    speak("Knowledge base loaded and indexed.")

# Query knowledge base

def query_knowledge_base(question):
    if kb_index is None:
        speak("Knowledge base is not loaded.")
        return
    q_embed = embedding_model.encode([question])
    D, I = kb_index.search(q_embed, k=3)
    answer = "\n".join([kb_chunks[i] for i in I[0]])
    speak("Here's what I found.")
    print(answer)
    speak(answer)

# Emotion Detection

def detect_emotion():
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)
    speak("Detecting emotion...")
    frame_count = 0
    detected_emotion = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = detector.detect_emotions(frame)
        if result:
            emotions = result[0]['emotions']
            detected_emotion = max(emotions, key=emotions.get)
            cv2.putText(frame, f"Emotion: {detected_emotion}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("EVA Emotion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or frame_count > 20:
            break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    if detected_emotion:
        speak(f"You seem {detected_emotion}.")

# Gesture Control

def gesture_control():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)
    speak("Gesture control activated. Press Q to stop.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]
                if abs(thumb.x - index.x) < 0.05 and abs(thumb.y - index.y) < 0.05:
                    pyautogui.click()
                    speak("Click gesture detected")
                elif index.y < hand_landmarks.landmark[6].y:
                    pyautogui.scroll(20)
                elif index.y > hand_landmarks.landmark[6].y:
                    pyautogui.scroll(-20)
        cv2.imshow("EVA Gesture", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    speak("Gesture control deactivated.")

# Commands

def execute_command(command):
    if "calculator" in command:
        os.system("calc")
        speak("Opening calculator")
    elif "notepad" in command:
        os.system("notepad")
        speak("Opening Notepad")
    elif "browser" in command:
        subprocess.run(["start", "chrome"], shell=True)
        speak("Opening browser")
    elif "screenshot" in command:
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        pyautogui.screenshot(filename)
        speak(f"Screenshot saved as {filename}")
    elif "emotion" in command:
        detect_emotion()
    elif "gesture" in command:
        gesture_control()
    elif "load knowledge" in command:
        speak("Please say the file path")
        file_path = listen()
        if os.path.exists(file_path):
            load_knowledge_file(file_path)
        else:
            speak("File not found")
    elif "ask knowledge" in command:
        speak("What is your question?")
        question = listen()
        if question:
            query_knowledge_base(question)
    elif "exit" in command:
        speak("Goodbye!")
        sys.exit()
    else:
        speak("Sorry, I didn’t understand that command.")

# Set user preferences

def set_preferences():
    global language_code
    speak("What is your name?")
    name = listen()
    if name:
        user_preferences['name'] = name
    speak("Which language should I speak? English, Hindi, Telugu, Spanish?")
    lang = listen()
    lang_map = {'english': 'en', 'hindi': 'hi', 'telugu': 'te', 'tamil': 'ta', 'spanish': 'es'}
    language_code = lang_map.get(lang.lower(), 'en')
    speak(f"Language set to {lang}.")

# Background loop

def eva_loop():
    speak(f"Hello {user_preferences['name']}, I am EVA.")
    dashboard.update_status("Listening...")
    while True:
        command = listen()
        if command:
            dashboard.update_command(command)
            dashboard.update_status("Processing...")
            execute_command(command)
            dashboard.update_status("Listening...")

# Main

def main():
    global dashboard
    set_preferences()
    dashboard = EvaDashboard()
    threading.Thread(target=eva_loop, daemon=True).start()
    dashboard.run()

if __name__ == "__main__":
    main()
