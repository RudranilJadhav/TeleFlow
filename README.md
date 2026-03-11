# 🏢 Stark Real Estate — Voice AI

**Smart Receptionist & Secretary for Real Estate**

A real-time, conversational Voice AI system designed to act as a **Smart Receptionist** (Inbound) and **Smart Secretary** (Outbound) for a real estate agency. It handles client inquiries, qualifies leads, supports barge-in (interruptions), and automatically generates **Minutes of Meeting (MoM)** with actionable insights.

---


https://github.com/user-attachments/assets/697e0c4d-3a6e-4a2b-917a-9268bf8dc00e



## ✨ Key Features

- **Real-Time Conversational AI** — Handles inbound & outbound phone calls with natural language.
- **Barge-in Support** — Silero VAD lets callers interrupt the AI mid-sentence, just like a human conversation.
- **Automated MoM Generation** — Transcribes calls with speaker diarization (Sarvam AI) and produces structured JSON MoMs (Groq LLM).
- **Analytics Dashboard** — Live Streamlit dashboard for active calls, MoM review, and lead metrics.
- **Ultra-Low Latency** — Faster-Whisper (ASR) + Groq (LLM) + Piper (TTS).

---

## 🏗️ Architecture & Tech Stack

| Layer | Technology |
|-------|-----------|
| **Telephony** | Asterisk PBX with ARI |
| **Speech-to-Text (Live)** | `faster-whisper` + `silero_vad` |
| **Speech-to-Text (MoM)** | Sarvam AI (`saaras:v3`) with diarization |
| **LLM** | Groq API (`llama-3.3-70b-versatile`) |
| **Text-to-Speech** | Piper TTS via UDP/RTP |
| **Audio Processing** | `ffmpeg`, `noisereduce`, `scipy` |
| **Dashboard** | Streamlit + Pandas |

---

## 📂 Project Structure

```
Devbits-PS-1/
├── src/
│   ├── services/
│   │   ├── main.py                    # Inbound call orchestrator (ASR + LLM + TTS)
│   │   ├── outbound.py                # Outbound call initiator
│   │   ├── brain.py                   # Groq LLM interface & conversation history
│   │   ├── events.py                  # Asterisk ARI event listener & bridge setup
│   │   ├── external_media.py          # RTP audio → VAD → Faster-Whisper
│   │   ├── tts.py                     # Piper TTS → RTP streaming
│   │   ├── vad_with_bargein.py        # Silero VAD with barge-in logic
│   │   ├── mom_generator.py           # Live call MoM generator
│   │   ├── generate_mom_from_audio.py # Batch MoM from recordings (Sarvam + Groq)
│   │   └── app.py                     # Streamlit analytics dashboard
│   ├── asterisk/                      # Asterisk PBX configuration files
│   └── piper-models/                  # Piper TTS voice models
├── call-recordings/                   # Raw audio files
├── audio-transcripts/                 # Sarvam AI diarized transcripts (JSON)
├── audio-mom/                         # Generated MoM files (JSON)
├── call-logs/                         # Call log data
├── requirements.txt
└── README.md
```

---

## ⚙️ Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.9+** | CUDA-enabled GPU recommended for Faster-Whisper |
| **Asterisk PBX** | With ARI enabled and ExternalMedia support |
| **FFmpeg** | Must be on system PATH |
| **Piper TTS** | Binary downloaded and accessible |
| **API Keys** | Groq + Sarvam AI (see below) |

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/RudranilJadhav/Devbits-PS-1.git
cd Devbits-PS-1
```

### 2. Create & activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Additional packages** for the MoM generator (Sarvam AI + Groq):
> ```bash
> pip install sarvamai groq python-dotenv
> ```

### 4. Install system dependencies

**FFmpeg** (required for audio processing):
```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

**Asterisk PBX** (required for telephony):
```bash
# Ubuntu / Debian
sudo apt install asterisk
```
> Copy the provided config files from `src/asterisk/` into your Asterisk config directory (usually `/etc/asterisk/`):
> ```bash
> sudo cp src/asterisk/*.conf /etc/asterisk/
> sudo systemctl restart asterisk
> ```

**Piper TTS**:
```bash
# Download the Piper binary for your platform from:
# https://github.com/rhasspy/piper/releases
#
# Place the binary somewhere on your PATH, e.g.:
wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz
tar -xzf piper_linux_x86_64.tar.gz
sudo mv piper /usr/local/bin/
```
> Voice models should be placed in `src/piper-models/`.

### 5. Set up environment variables

Create a `.env` file inside `src/services/`:

```bash
cat > src/services/.env << 'EOF'
GROQ_API_KEY=your_groq_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
EOF
```

| Key | Get it from |
|-----|-------------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/) |
| `SARVAM_API_KEY` | [dashboard.sarvam.ai](https://dashboard.sarvam.ai/) |

### 6. Configure Asterisk ARI

Ensure `src/asterisk/ari.conf` has a valid ARI user:

```ini
[general]
enabled = yes

[asterisk]
type = user
password = asterisk
password_format = plain
```

---

## ▶️ Running

### Start the Inbound AI Receptionist

```bash
cd src/services
python main.py
```

### Start an Outbound Call

```bash
cd src/services
python outbound.py
```

### Generate MoM from Recorded Audio

```bash
cd src/services

# Single file
python generate_mom_from_audio.py ../../call-recordings/my_call.mp3

# Interactive directory picker
python generate_mom_from_audio.py --dir ../../call-recordings
```

### Launch the Analytics Dashboard

```bash
cd src/services
streamlit run app.py
```

---

## 📊 MoM Output Schema

The MoM generator produces a structured JSON for every call:

```json
{
  "meeting_title": "Lead Inquiry — 3BHK in Pune",
  "participants": ["Speaker 0", "Speaker 1"],
  "summary": "Caller inquired about 3BHK apartments...",
  "key_discussion_points": [
    {
      "topic": "Budget",
      "details": "Caller's budget is 80L–1Cr",
      "raised_by": "Speaker 0"
    }
  ],
  "decisions_made": ["Site visit scheduled for Saturday"],
  "action_items": [
    {
      "task": "Send brochure for Project X",
      "assignee": "Speaker 1",
      "deadline": "Tomorrow"
    }
  ],
  "follow_up": "Call back after site visit",
  "sentiment": "Positive",
  "_meta": {
    "source_file": "call_20260311",
    "stt": "sarvam_saaras_v3",
    "llm": "groq_llama-3.3-70b"
  }
}
```

---

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
