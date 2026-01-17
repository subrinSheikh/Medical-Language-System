import os
import time
import json
from datetime import datetime
os.environ["PYTORCH_DISABLE_TORCH_LOAD_SECURITY_CHECK"] = "1"

from flask import Flask, render_template, request
import whisper
from gtts import gTTS
from deep_translator import GoogleTranslator
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ================= CONFIG =================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Warning: Failed to initialize Gemini client: {e}")
        client = None
else:
    print("Warning: GEMINI_API_KEY not found in environment variables")

# ===== Rate limit safety =====
LAST_GEMINI_CALL = 0
GEMINI_COOLDOWN = 2  # seconds

def can_call_gemini():
    global LAST_GEMINI_CALL
    now = time.time()
    if now - LAST_GEMINI_CALL < GEMINI_COOLDOWN:
        return False
    LAST_GEMINI_CALL = now
    return True

# ================= APP =================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
HISTORY_FILE = "history.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= HISTORY FUNCTIONS =================
def load_history():
    """Load history from JSON file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(entry):
    """Save a new entry to history"""
    history = load_history()
    history.insert(0, entry)  # Add to beginning
    # Keep only last 50 entries
    history = history[:50]
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

def detect_language(text):
    """Detect the language of the input text"""
    try:
        from langdetect import detect
        lang_code = detect(text)
        lang_map = {
            "en": "English", "hi": "Hindi", "ur": "Urdu",
            "ar": "Arabic", "bn": "Bengali", "ne": "Nepali",
            "ja": "Japanese", "zh": "Chinese", "zh-cn": "Chinese"
        }
        return lang_map.get(lang_code, "Unknown")
    except:
        return "Auto Detect"

def get_flag_emoji(language):
    """Get flag emoji for language"""
    flag_map = {
        "English": "🇬🇧", "Hindi": "🇮🇳", "Urdu": "🇵🇰",
        "Arabic": "🇸🇦", "Bengali": "🇧🇩", "Nepali": "🇳🇵",
        "Japanese": "🇯🇵", "Chinese": "🇨🇳", "French": "🇫🇷",
        "Spanish": "🇪🇸", "German": "🇩🇪", "Portuguese": "🇵🇹",
        "Unknown": "🌐", "Auto Detect": "🌐"
    }
    return flag_map.get(language, "🌐")

# Jinja2 filter for flag emoji (must be after get_flag_emoji function)
@app.template_filter('get_flag')
def get_flag_filter(language):
    return get_flag_emoji(language)

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper loaded.")

# ================= AI TUTOR (UNCHANGED) =================
def ai_tutor_response(question):
    if not client:
        return "Error: Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
    
    if not can_call_gemini():
        return "Please wait a few seconds before asking again."

    try:
        prompt = f"""
You are a medical AI tutor.
Answer clearly and simply.
DO NOT diagnose.
Question: {question}
"""
        response = client.models.generate_content(
            model="models/gemini-flash-lite-latest",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating AI tutor response: {str(e)}"

# ================= EXPLAIN CONDITION =================
def explain_condition(condition_text):
    """Explain medical condition in simple words"""
    if not client:
        return {
            "what_it_means": "Error: Gemini API key not configured.",
            "what_to_do": "",
            "what_not_to_do": ""
        }
    
    if not can_call_gemini():
        return {
            "what_it_means": "Please wait a few seconds before asking again.",
            "what_to_do": "",
            "what_not_to_do": ""
        }

    try:
        prompt = f"""
You are a medical AI assistant. A user has described a medical condition: "{condition_text}"

Explain this in simple, clear language. Structure your response in three sections:

1. WHAT IT COULD MEAN: Explain what this condition/symptom might indicate in simple terms (2-3 sentences)

2. WHAT TO DO NOW: Provide immediate actionable steps the person should take (3-4 bullet points)

3. WHAT NOT TO DO: List things they should avoid doing (2-3 bullet points)

IMPORTANT: 
- DO NOT diagnose or provide specific medical diagnosis
- Use simple, non-technical language
- Be empathetic and clear
- Focus on general guidance, not specific treatment

Format your response exactly as:
WHAT IT COULD MEAN: [explanation]
WHAT TO DO NOW: [steps]
WHAT NOT TO DO: [warnings]
"""
        response = client.models.generate_content(
            model="models/gemini-flash-lite-latest",
            contents=prompt
        )
        
        text = response.text.strip()
        
        # Parse the response
        result = {
            "what_it_means": "",
            "what_to_do": "",
            "what_not_to_do": ""
        }
        
        if "WHAT IT COULD MEAN:" in text:
            parts = text.split("WHAT IT COULD MEAN:")
            if len(parts) > 1:
                rest = parts[1]
                if "WHAT TO DO NOW:" in rest:
                    result["what_it_means"] = rest.split("WHAT TO DO NOW:")[0].strip()
                    rest = rest.split("WHAT TO DO NOW:")[1]
                    if "WHAT NOT TO DO:" in rest:
                        result["what_to_do"] = rest.split("WHAT NOT TO DO:")[0].strip()
                        result["what_not_to_do"] = rest.split("WHAT NOT TO DO:")[1].strip()
                    else:
                        result["what_to_do"] = rest.strip()
                else:
                    result["what_it_means"] = rest.strip()
        else:
            # Fallback: return entire text
            result["what_it_means"] = text
        
        return result
    except Exception as e:
        return {
            "what_it_means": f"Error: {str(e)}",
            "what_to_do": "",
            "what_not_to_do": ""
        }

# ================= SILENT EMERGENCY MODE =================
EMERGENCY_MESSAGES = {
    "chest_pain": {
        "en": "I am experiencing chest pain. This is a medical emergency. Please call an ambulance immediately.",
        "icon": "🫀",
        "title": "Chest Pain"
    },
    "head_injury": {
        "en": "I have a head injury. I need immediate medical attention. Please help me get to a hospital.",
        "icon": "🤕",
        "title": "Head Injury"
    },
    "dizziness": {
        "en": "I am feeling dizzy and cannot speak clearly. I need medical help. Please assist me.",
        "icon": "😵",
        "title": "Dizziness"
    },
    "breathing": {
        "en": "I am having difficulty breathing. This is an emergency. Please call for medical help immediately.",
        "icon": "😮‍💨",
        "title": "Breathing Difficulty"
    },
    "stroke": {
        "en": "I think I am having a stroke. I cannot speak properly. Please call an ambulance right away.",
        "icon": "🧠",
        "title": "Possible Stroke"
    },
    "choking": {
        "en": "I am choking and cannot speak. I need immediate help. Please perform the Heimlich maneuver.",
        "icon": "😰",
        "title": "Choking"
    },
    "abuse": {
        "en": "I am in danger and cannot speak. I need help. Please contact emergency services.",
        "icon": "🆘",
        "title": "Emergency Help"
    },
    "allergy": {
        "en": "I am having a severe allergic reaction. I need an epinephrine injection and immediate medical care.",
        "icon": "🤧",
        "title": "Allergic Reaction"
    }
}

def get_emergency_message(emergency_type, language):
    """Get emergency message in specified language"""
    if emergency_type not in EMERGENCY_MESSAGES:
        return None
    
    message_en = EMERGENCY_MESSAGES[emergency_type]["en"]
    
    if language == "English":
        return message_en
    else:
        return translate_text(message_en, language)

# ================= TRANSLATOR EMOTION (MODEL-BASED) =================
def detect_emotion_for_translator(text):
    if not client:
        return "neutral"
    
    if not can_call_gemini():
        return "neutral"

    try:
        prompt = f"""
Detect the emotion of the following medical text.

Reply with ONLY one word from:
happy, sad, anxious, urgent, calm, excited, neutral

Text:
{text}
"""

        response = client.models.generate_content(
            model="models/gemini-flash-lite-latest",
            contents=prompt
        )
        return response.text.strip().lower()
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return "neutral"

# ================= TRANSLATION =================
def translate_text(text, language):
    lang_map = {
        "English": "en", "Hindi": "hi", "Urdu": "ur",
        "Arabic": "ar", "Bengali": "bn",
        "Nepali": "ne", "Japanese": "ja", "Chinese": "zh"
    }

    try:
        return GoogleTranslator(
            source="auto",
            target=lang_map.get(language, "en")
        ).translate(text)
    except Exception as e:
        print(f"Error translating text: {e}")
        return f"Translation error: {str(e)}"

# ================= TTS =================
def generate_tts(text, language, emergency=False):
    lang_map = {
        "English": "en", "Hindi": "hi", "Urdu": "ur",
        "Arabic": "ar", "Bengali": "bn",
        "Nepali": "ne", "Japanese": "ja", "Chinese": "zh"
    }

    try:
        prefix = "Emergency alert. " if emergency else ""
        tts = gTTS(
            prefix + text,
            lang=lang_map.get(language, "en"),
            slow=emergency
        )
        os.makedirs("static", exist_ok=True)
        tts.save("static/output.mp3")
    except Exception as e:
        print(f"Error generating TTS: {e}")

# ================= ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():

    recognized = None
    translated = None
    emotion = None
    tutor_response = None
    tutor_audio = False

    if request.method == "POST":
        mode = request.form.get("mode")

        # ========== AI TUTOR ==========
        if mode == "tutor":
            question = request.form.get("tutor_query")
            language = request.form.get("language", "English")

            if question:
                answer_en = ai_tutor_response(question)
                tutor_response = translate_text(answer_en, language)
                generate_tts(tutor_response, language, emergency=False)
                tutor_audio = True

                # Save to history
                history_entry = {
                    "type": "tutor",
                    "question": question,
                    "response": tutor_response,
                    "language": language,
                    "timestamp": datetime.now().isoformat()
                }
                save_history(history_entry)

            history = load_history()
            return render_template(
                "index.html",
                active_tab="tutor",
                tutor_response=tutor_response,
                tutor_audio=tutor_audio,
                history=history
            )

        # ========== EXPLAIN CONDITION ==========
        if mode == "explain_condition":
            condition = request.form.get("condition_text")
            language = request.form.get("language", "English")
            
            explanation = None
            explain_audio = False
            
            if condition:
                explanation = explain_condition(condition)
                
                # Translate explanation
                if explanation["what_it_means"]:
                    explanation_text = f"What it could mean: {explanation['what_it_means']}\n\nWhat to do now: {explanation['what_to_do']}\n\nWhat not to do: {explanation['what_not_to_do']}"
                    explanation_translated = translate_text(explanation_text, language)
                    generate_tts(explanation_translated, language, emergency=True)
                    explain_audio = True
                    
                    # Save to history
                    history_entry = {
                        "type": "explain_condition",
                        "condition": condition,
                        "explanation": explanation,
                        "language": language,
                        "timestamp": datetime.now().isoformat()
                    }
                    save_history(history_entry)
            
            history = load_history()
            return render_template(
                "index.html",
                active_tab="explain_condition",
                explanation=explanation,
                condition_text=condition,
                explain_audio=explain_audio,
                history=history
            )

        # ========== SILENT EMERGENCY MODE ==========
        if mode == "silent_emergency":
            emergency_type = request.form.get("emergency_type")
            language = request.form.get("language", "English")
            
            emergency_message = None
            emergency_audio = False
            emergency_info = None
            
            if emergency_type:
                emergency_message = get_emergency_message(emergency_type, language)
                emergency_info = EMERGENCY_MESSAGES.get(emergency_type, {})
                
                if emergency_message:
                    # Generate TTS with emergency alert
                    generate_tts(emergency_message, language, emergency=True)
                    emergency_audio = True
                    
                    # Save to history
                    history_entry = {
                        "type": "silent_emergency",
                        "emergency_type": emergency_type,
                        "message": emergency_message,
                        "language": language,
                        "timestamp": datetime.now().isoformat()
                    }
                    save_history(history_entry)
            
            history = load_history()
            return render_template(
                "index.html",
                active_tab="silent_emergency",
                emergency_message=emergency_message,
                emergency_type=emergency_type,
                emergency_info=emergency_info,
                emergency_audio=emergency_audio,
                history=history
            )

        # ========== TRANSLATOR ==========
        text_input = request.form.get("text_input")
        audio = request.files.get("audio")
        language = request.form.get("language")

        if text_input and text_input.strip():
            recognized = text_input
        elif audio:
            path = os.path.join(UPLOAD_FOLDER, "input.wav")
            audio.save(path)
            recognized = whisper_model.transcribe(path)["text"]

        if recognized:
            emotion = detect_emotion_for_translator(recognized)
            emergency = emotion in ["urgent", "anxious"]

            # Detect source language
            source_lang = detect_language(recognized)
            
            translated = translate_text(recognized, language)
            generate_tts(translated, language, emergency)

            # Save to history
            history_entry = {
                "type": "translation",
                "source_text": recognized,
                "translated_text": translated,
                "source_language": source_lang,
                "target_language": language,
                "emotion": emotion,
                "timestamp": datetime.now().isoformat()
            }
            save_history(history_entry)

            history = load_history()
            return render_template(
                "index.html",
                active_tab="translator",
                recognized=recognized,
                translated=translated,
                emotion=emotion,
                history=history
            )

    history = load_history()
    return render_template("index.html", active_tab="translator", history=history)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True, port=5001)
