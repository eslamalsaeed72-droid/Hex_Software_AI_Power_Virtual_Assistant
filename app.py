import streamlit as st
import sqlite3
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from langdetect import detect, LangDetectException
from dateutil.parser import parse
import re
import os
from streamlit_mic_recorder import speech_to_text

# ملف قاعدة البيانات
DB_FILE = "reminders.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reminders
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  task TEXT NOT NULL,
                  time TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

# تحميل النماذج
@st.cache_resource
def load_english_classifier():
    return pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                    device=0 if torch.cuda.is_available() else -1)

english_classifier = load_english_classifier()

@st.cache_resource
def load_arabic_model():
    model_path = "arabic_finetuned_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

arabic_tokenizer, arabic_model = load_arabic_model()

candidate_labels = [
    "greeting",
    "add_reminder",
    "view_reminders",
    "general_question"
]

# إدارة اللغة
if "language" not in st.session_state:
    st.session_state.language = "ar"

if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome = "مرحبا! كيف يمكنني مساعدتك اليوم؟" if st.session_state.language == "ar" else "Hello! How can I help you today?"
    st.session_state.messages.append({"role": "assistant", "content": welcome})

# دوال مساعدة
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'en'

def get_text(key, **kwargs):
    texts = {
        "title": {"ar": "المساعد الذكي", "en": "AI Assistant"},
        "caption": {"ar": "مساعد ذكي لإدارة التذكيرات", "en": "Smart Reminder Assistant"},
        "input_placeholder": {"ar": "اكتب رسالتك هنا...", "en": "Type your message here..."},
        "voice_start": {"ar": "ابدأ التسجيل 🎤", "en": "Start Recording 🎤"},
        "voice_stop": {"ar": "إيقاف ⏹", "en": "Stop ⏹"},
        "greeting": {"ar": "مرحبا! ازيك؟", "en": "Hello! How are you?"},
        "not_understood": {"ar": "معلش، ما فهمتش. ممكن تكرر؟", "en": "Sorry, didn't understand. Rephrase?"},
        "add_success": {"ar": "تم إضافة: **{task}** في {time}", "en": "Added: **{task}** at {time}"},
        "no_reminders": {"ar": "ما عندكش تذكيرات", "en": "No reminders yet"},
        "reminders_title": {"ar": "تذكيراتك:", "en": "Your Reminders:"},
        "language_changed": {"ar": "تم تغيير اللغة إلى العربية", "en": "Language changed to English"},
        "clear_all": {"ar": "مسح كل التذكيرات", "en": "Clear All Reminders"},
        "cleared": {"ar": "تم مسح كل التذكيرات", "en": "All reminders cleared"}
    }
    lang = st.session_state.language
    txt = texts.get(key, {}).get(lang, "")
    return txt.format(**kwargs) if kwargs else txt

def clear_all_reminders():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM reminders")
    conn.commit()
    conn.close()

def process_message(message):
    detected_lang = detect_language(message)

    if detected_lang == 'ar':
        inputs = arabic_tokenizer(message, return_tensors="pt", truncation=True, max_length=128)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            logits = arabic_model(**inputs).logits
        intent = arabic_model.config.id2label[logits.argmax(-1).item()]
        conf = torch.softmax(logits, -1)[0][logits.argmax(-1)].item()
    else:
        res = english_classifier(message, candidate_labels, multi_label=False)
        intent = res["labels"][0]
        conf = res["scores"][0]

    if conf < 0.35:
        return get_text("not_understood")

    if intent == "greeting":
        return get_text("greeting")

    elif intent == "add_reminder":
        task = message
        time_str = "غير محدد"

        try:
            parsed = parse(message, fuzzy=True, dayfirst=True)
            time_str = parsed.strftime("%Y-%m-%d %H:%M")
            task = re.sub(r'\d{4}.*\d{2}:\d{2}', '', message).strip() or "مهمة"
        except:
            pass

        conn = sqlite3.connect(DB_FILE)
        conn.execute("INSERT INTO reminders (task, time) VALUES (?, ?)", (task, time_str))
        conn.commit()
        conn.close()

        return get_text("add_success", task=task, time=time_str)

    elif intent == "view_reminders":
        conn = sqlite3.connect(DB_FILE)
        reminders = conn.execute("SELECT id, task, time FROM reminders").fetchall()
        conn.close()
        if not reminders:
            return get_text("no_reminders")
        lines = [get_text("reminders_title")]
        for rid, task, tm in reminders:
            lines.append(f"#{rid} - {task} @ {tm}")
        return "\n".join(lines)

    return get_text("not_understood")

# واجهة التطبيق
st.set_page_config(page_title="المساعد الذكي", layout="wide")

st.title(get_text("title"))
st.caption(get_text("caption"))

# زر تغيير اللغة في الـ sidebar (طريقة مضمونة)
with st.sidebar:
    st.header("الإعدادات" if st.session_state.language == "ar" else "Settings")
    if st.button("تغيير اللغة / Change Language"):
        # تبديل اللغة
        st.session_state.language = "en" if st.session_state.language == "ar" else "ar"
        st.rerun()  # إعادة تحميل كامل الواجهة

    if st.button(get_text("clear_all")):
        clear_all_reminders()
        st.sidebar.success(get_text("cleared"))
        st.rerun()

# أزرار سريعة (بدون حذف فردي)
quick_options = {
    "ar": ["أضف تذكير", "عرض التذكيرات"],
    "en": ["Add Reminder", "Show Reminders"]
}

st.markdown("### أوامر سريعة" if st.session_state.language == "ar" else "### Quick Actions")
cols = st.columns(2)
for i, label in enumerate(quick_options[st.session_state.language]):
    if cols[i].button(label, key=f"quick_{i}", use_container_width=True):
        st.session_state["current_prompt"] = label
        st.rerun()

# عرض الرسائل
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# معالجة الإدخال
if "current_prompt" in st.session_state:
    prompt = st.session_state.pop("current_prompt")
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("جاري..."):
            response = process_message(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

if prompt := st.chat_input(get_text("input_placeholder")):
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("جاري..."):
            response = process_message(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# الإدخال الصوتي
st.markdown("### أو استخدم الصوت" if st.session_state.language == "ar" else "### Or use voice")
voice_text = speech_to_text(
    language=st.session_state.language,
    start_prompt=get_text("voice_start"),
    stop_prompt=get_text("voice_stop"),
    just_once=True,
    key="voice_input"
)

if voice_text:
    st.chat_message("user").markdown(voice_text)
    with st.chat_message("assistant"):
        with st.spinner("جاري..."):
            response = process_message(voice_text)
        st.markdown(response)
    st.session_state.messages.append({"role": "user", "content": voice_text})
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()