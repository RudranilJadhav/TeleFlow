# Devbits-PS-1
# 🏢 Stark Real Estate - Voice AI 

**Smart Receptionist & Secretary for Real Estate**

A real-time, conversational Voice AI system designed to act as a Smart Receptionist (Inbound) and Smart Secretary (Outbound) for a real estate agency. Built for low latency and natural conversational flow, it handles client inquiries, qualifies leads, gracefully handles interruptions (barge-in), and automatically generates Minutes of Meeting (MoM) with actionable insights.

---

## ✨ Key Features

* **Real-Time Conversational AI:** Handles inbound and outbound phone calls using natural language.
* **Barge-in Support:** Advanced Voice Activity Detection (VAD) allows callers to interrupt the AI seamlessly, mimicking human conversation.
* **Automated MoM Generation:** Automatically transcribes and analyzes calls to extract lead quality, budgets, preferred cities, and configurations, saving them as structured documents.
* **Real Estate Analytics Dashboard:** A live Streamlit dashboard to monitor active calls, review MoMs, and visualize lead metrics (Hot/Warm/Cold, Budget Distributions, City Demands).
* **Ultra-Low Latency Stack:** Powered by Faster-Whisper (ASR), Groq (Llama-3.3-70b-versatile LLM), and Piper (TTS).

---

## 🏗️ Architecture & Tech Stack

* **Telephony Engine:** Asterisk PBX with ARI (Asterisk REST Interface)
* **Speech-to-Text (ASR):** `faster-whisper` with `silero_vad` for voice activity and barge-in detection.
* **LLM Brain:** Groq API (`llama-3.3-70b-versatile`)
* **Text-to-Speech (TTS):** Piper TTS via UDP/RTP streaming
* **Audio Processing:** `ffmpeg` for transcoding RTP audio streams, `noisereduce` & `scipy` for filtering.
* **Dashboard:** Streamlit & Pandas

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `main.py` | Main entry point for **Inbound** calls. Orchestrates ASR, LLM, TTS, and Asterisk Events via multiprocessing. |
| `outbound.py` | Entry point for **Outbound** calls. Initiates the call via ARI and starts the AI pipeline. |
| `app.py` | Streamlit Dashboard for Live Call Monitoring and MoM Analytics. |
| `brain.py` | Interfaces with the Groq API. Manages conversation history and streaming LLM responses. |
| `events.py` | Asterisk ARI event listener. Sets up mixing bridges, snooping channels, and triggers MoM generation on call end. |
| `external_media.py` | Handles incoming RTP audio from Asterisk, processes it through VAD, and runs Faster-Whisper for transcription. |
| `tts.py` | Converts LLM text output into speech using Piper TTS and streams it back to Asterisk via RTP. |
| `vad_with_bargein.py` | Custom Silero VAD implementation handling speech thresholds, silences, backchannels, and interruption logic. |
| `mom_generator.py` | Uses Groq to parse call transcripts into structured JSON (budget, intent, etc.) and generates a formatted text MoM. |

---

## ⚙️ Prerequisites

1.  **Asterisk PBX**: Configured with ARI enabled and external media support.
2.  **FFmpeg**: Installed and added to system PATH.
3.  **Piper TTS**: Piper binary downloaded and accessible.
4.  **Python 3.9+**: With a CUDA-enabled GPU recommended for Faster-Whisper.
5.  **API Keys**: A valid [Groq API Key](https://console.groq.com/).

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
