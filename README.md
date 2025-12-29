
# Hex Software AI Powered Virtual Assistant

A smart, multilingual virtual assistant built with Streamlit that understands both **Arabic** and **English**, manages reminders, supports voice input, and uses fine-tuned NLP models for accurate intent detection.

## Features

- Full bilingual support (Arabic ↔ English) with instant language switching
- Reminder system:
  - Add new reminders (text or voice)
  - View all saved reminders
  - Clear all reminders with one click
- Voice input using real-time speech-to-text (via `streamlit-mic-recorder`)
- Intelligent intent classification:
  - English: zero-shot with mDeBERTa-v3
  - Arabic: custom fine-tuned model based on AraBERTv2
- Clean chat-style interface with quick action buttons
- Persistent storage using SQLite database

## Tech Stack

- **Core Framework**: Streamlit
- **NLP**:
  - Transformers (Hugging Face)
  - Fine-tuned AraBERTv2 (Arabic)
  - mDeBERTa-v3-base-mnli-xnli (English zero-shot)
- **Speech-to-Text**: streamlit-mic-recorder
- **Database**: SQLite
- **Utilities**: langdetect, python-dateutil

## Project Structure

```
Hex_Software_AI_Power_Virtual_Assistant/
├── app.py                        # Main Streamlit application
├── arabic_finetuned_model/       # Fine-tuned Arabic intent classification model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
├── assistant_model.ipynb         # Jupyter Notebook: Arabic model training & fine-tuning
├── reminders.db                  # SQLite database (auto-generated)
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Installation

1. Clone the repository

```bash
git clone https://github.com/eslamalsaeed72-droid/Hex_Software_AI_Power_Virtual_Assistant.git
cd Hex_Software_AI_Power_Virtual_Assistant
```

2. Create & activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install requirements

```bash
pip install -r requirements.txt
```

4. Run the app

```bash
streamlit run app.py
```

## Requirements

```text
streamlit>=1.31.0
transformers>=4.35.0
torch>=2.0.0
langdetect>=1.0.9
python-dateutil>=2.8.2
streamlit-mic-recorder>=0.0.8
```

> **Important Note**:  
> The complete training pipeline for the Arabic fine-tuned model (including data preparation from MASSIVE dataset, intent mapping, training, and saving) is available in:  
> **`assistant_model.ipynb`**

## How to Use

- **Text commands**  
  - "ذكرني أروح السوبر ماركت 5 مساء"  
  - "عرض التذكيرات"  
  - "Remind me to call Ahmed tomorrow at 3 PM"  
  - "Show reminders"

- **Voice input**  
  Click the microphone icon and speak naturally

- **Switch language**  
  Use the sidebar button or type:  
  "غير اللغة" / "change to English"

- **Clear all reminders**  
  Use the sidebar button

## Model Training

The Arabic intent classification model was fine-tuned using:
- **Dataset**: MASSIVE (Arabic - ar-SA locale)
- **Base model**: aubmindlab/bert-base-arabertv2
- **Notebook**: `assistant_model.ipynb`  
  Contains full pipeline: data loading, custom intent mapping, tokenization, training with Hugging Face Trainer, evaluation, and model saving

## Future Plans

- Google Calendar integration
- Reminder editing & individual deletion
- Enhanced time & entity extraction
- Dark mode & UI themes
- Export/import reminders feature

## License

MIT License

Copyright © 2025 Eslam AlSaeed

---

**Tags / Keywords**  
`python` `streamlit` `ai-assistant` `virtual-assistant` `multilingual` `arabic-nlp` `voice-assistant` `speech-to-text` `intent-classification` `fine-tuning` `transformers` `bert` `arbert` `reminder-app` `sqlite`
```
