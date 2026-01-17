# 🩺 Medical Emergency Translation System

An AI-powered multilingual medical assistance system designed to help users communicate medical information across language barriers, understand medical conditions, and get emergency help — even when they cannot speak.

---

## 🚀 Project Overview

The **Medical Emergency Translation System** is a smart AI application that combines:
- Speech recognition
- Language translation
- Emotion detection
- Text-to-speech
- Large Language Models (LLMs)

It is especially useful in **medical emergencies**, **travel scenarios**, and **healthcare accessibility** contexts.

---

## 🎯 Key Features

### 🔹 Translator with Emotion Detection
- Translates medical text or speech into multiple languages
- Automatically detects **user emotion** (urgent, anxious, calm, sad, etc.)
- Adjusts voice output style for emergency situations

### 🔹 AI Medical Tutor (Voice Enabled)
- Ask medical questions in natural language
- AI responds with **simple, non-diagnostic explanations**
- Output is available in **text + voice** in selected language

### 🔹 Explain My Condition
- User describes symptoms or condition
- AI explains:
  - What it could mean
  - What to do now
  - What not to do
- Designed for non-technical users

### 🔹 Silent Emergency Mode
- For users who cannot speak
- Tap icons (chest pain, stroke, breathing difficulty, etc.)
- Generates and speaks emergency messages instantly in target language

### 🔹 History Tracking
- Saves recent translations, tutor queries, and emergency actions
- Helps review past interactions

---

## 🧠 AI & Technologies Used

- **Whisper (OpenAI)** – Speech-to-text
- **Google Gemini (LLM)** – AI tutor, emotion detection, condition explanation
- **Deep Translator** – Language translation
- **gTTS** – Text-to-speech voice output
- **Flask** – Backend web framework
- **HTML / CSS / JavaScript** – Frontend UI
- **JSON** – History storage

---

## 🏗 System Architecture (High Level)

1. User inputs text or voice
2. Speech → Text (Whisper)
3. Emotion detection (LLM-based)
4. Translation or AI reasoning (Gemini)
5. Text-to-Speech output
6. Results displayed + audio playback

---

👩‍💻 Author

Project: Medical Emergency Translation System
Developed by: [Subrin Sheikh]

## 📂 Project Structure

```text
medical_translation_ai/
│
├── app.py                 # Main Flask backend
├── templates/
│   └── index.html         # UI template
├── static/
│   ├── output.mp3         # Generated speech output
│   └── background.avif    # UI background
├── uploads/               # Temporary audio uploads
├── history.json           # Interaction history
├── notebook/
│   └── Medical_Emergency_Translation_System.ipynb
├── README.md              # Project documentation
├── .env                   # API keys (NOT uploaded)
└── requirements.txt



