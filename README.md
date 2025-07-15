# 🤖 EVA - Enhanced Voice Assistant

EVA is an AI-powered voice assistant built entirely in Python that works **offline** and integrates multiple powerful features:
- 🎛️ GUI Dashboard
- ✋ Real-Time Gesture Control
- 🌍 Multilingual Support
- 😊 Facial Recognition & Emotion Detection
- 📚 Offline Knowledge Base with Embedding Search

---

## 🚀 Features

| Feature                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| GUI Dashboard                    | Tkinter-based control panel for EVA's status, logs, and controls            |
| Voice Recognition & Commands     | Use your voice to control apps, websites, system tools, and more           |
| Real-Time Gesture Control        | MediaPipe-based gesture detection for scrolling and clicking                |
| Multilingual Support             | Input/output language translation using Google Translate                    |
| Facial Recognition & Emotion AI | Webcam-based emotion detection using FER and dynamic response adaptation    |
| Offline Knowledge Base           | Query PDF/text documents using FAISS + Sentence Transformers                |

---

## 🧰 Installation

### ✅ Requirements
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 📦 Required Libraries
```
pyttsx3
SpeechRecognition
spacy
wikipedia
pyautogui
pyaudio
googletrans==4.0.0-rc1
opencv-python
mediapipe
fer
faiss-cpu
sentence-transformers
PyMuPDF
tkinter (included with Python)
```

---

## 💻 Running EVA

```bash
python eva.py
```

EVA will launch a GUI, ask your name, preferred language, and begin listening for voice commands.

---

## 🎤 Sample Voice Commands

| Category           | Examples                                           |
|--------------------|----------------------------------------------------|
| System Apps        | "open notepad", "open calculator"                  |
| Web & Wikipedia    | "search browser", "search wikipedia python"        |
| Gesture Control    | "start gesture control" (use hand to scroll/click)|
| Emotion Detection  | "detect my emotion"                                |
| Offline QA         | "load knowledge base", "ask knowledge base"        |
| Multilingual       | Set preferred language to Hindi, Telugu, etc.      |
| Exit               | "exit", or press Exit on GUI                       |

---

## 🧠 Offline Document QA (FAISS)
1. Say: **"load knowledge base"** → provide PDF/TXT path
2. Say: **"ask knowledge base"** → ask your question
3. EVA responds from local content without internet

---

## 🧪 Real-Time Gestures
| Gesture         | Action             |
|-----------------|--------------------|
| ✋ Open hand     | Activate gesture   |
| 👉 Point up/down | Scroll content     |
| 🤏 Pinch fingers | Click              |

---

## 😊 Emotion Detection
EVA uses your webcam to detect:
- happy, sad, angry, neutral, surprise, etc.
And responds empathetically.

---

## 📁 Project Structure
```
EVA/
├── eva.py              # Main assistant script (full integration)
├── requirements.txt    # All dependencies
├── assistant.log       # Activity log
├── README.md           # You're reading it
├── User_Manual.md      # Full usage guide (optional)
```

---

## 📜 License
**MIT License** — free to use, modify, and distribute with credit.

---

## 👨‍💻 Author
Developed by **YOU** using Python & open-source libraries.

---

## 📣 Coming Soon
- 🎵 Media control
- 💬 Chat with EVA (LLM)
- 🧩 Plugin system
- 🔐 User authentication

---
