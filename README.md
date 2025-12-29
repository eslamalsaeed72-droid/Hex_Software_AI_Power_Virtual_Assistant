
# AI-Powered Virtual Assistant

A multilingual intelligent virtual assistant built with Streamlit, supporting both **Arabic** and **English**.  
The system can understand natural language commands, manage reminders (add, view, clear all), accept voice input, and classify user intents using fine-tuned language models.

## Features

- Bilingual interface (Arabic ↔ English) with automatic language detection and manual switching
- Reminder management:
  - Add new reminders (text or voice)
  - View all saved reminders
  - Clear all reminders with one click
- Real-time voice input with speech-to-text (supports Arabic & English)
- Advanced intent classification:
  - Zero-shot classification for English (mDeBERTa-v3)
  - Fine-tuned model for Arabic (based on AraBERTv2)
- Clean, responsive chat interface with quick action buttons
- Persistent storage using SQLite

## Project Structure

```
AI_Powered_Virtual_Assistant/
├── app.py                        # Main Streamlit application
├── arabic_finetuned_model/       # Fine-tuned Arabic intent classification model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
├── assistant_model.ipynb         # Jupyter Notebook containing the training & fine-tuning code for the Arabic model
├── reminders.db                  # SQLite database (auto-created)
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## Installation

1. Clone the repository

```bash
git clone (https://github.com/eslamalsaeed72-droid/Hex_Software_AI_Power_Virtual_Assistant)
```

2. Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the application

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

> **Note**: The fine-tuning process for the Arabic model is fully documented and executable in the file `assistant_model.ipynb`.  
> It includes data loading from MASSIVE dataset (Arabic subset), intent mapping, tokenization, training with Trainer API, and model saving.

## Usage Examples

- **Text commands**  
  - "ذكرني أروح السوبر ماركت 5 مساء"  
  - "عرض التذكيرات"  
  - "Remind me to call Ahmed tomorrow at 3 PM"  
  - "Show reminders"

- **Voice commands**  
  Click the microphone icon and speak naturally in Arabic or English

- **Language switching**  
  Type "غير اللغة" or "change to English"  
  Or use the sidebar button

## Development & Training

The Arabic intent classification model was fine-tuned using:
- **Dataset**: MASSIVE (Arabic subset - ar-SA locale)
- **Base model**: aubmindlab/bert-base-arabertv2
- **Training notebook**: `assistant_model.ipynb`  
  Contains complete pipeline: data loading, intent mapping to custom labels, tokenization, training arguments, and model saving

## Future Enhancements (Roadmap)

- Google Calendar integration
- Reminder editing & deletion by ID
- Improved time parsing with better NLP
- Dark mode & custom themes
- Export/import reminders feature

## License

MIT License

Copyright (c) 2025 eslamalsaeed72-droid

---

Keywords / Tags:  
`python` `streamlit` `artificial-intelligence` `natural-language-processing` `voice-assistant` `multilingual` `arabic-nlp` `english-nlp` `reminder-app` `speech-to-text` `intent-classification` `fine-tuning` `transformers` `bert` `arbert` `mdeberta` `sqlite`
```

This README is now complete, professional, well-structured, and includes everything you asked for.  
You can copy-paste it directly into your `README.md` file.  

Good luck with your project showcase!
