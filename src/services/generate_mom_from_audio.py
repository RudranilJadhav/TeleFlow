"""
Generate a structured JSON MoM from a saved call recording.

Usage:
    python generate_mom_from_audio.py <path_to_audio_file>

Pipeline:
    1. Groq Whisper (whisper-large-v3) — multilingual speech-to-text
    2. Groq LLM (llama-3.3-70b-versatile) — structured MoM extraction
    3. Output — JSON saved alongside the audio + printed to stdout
"""

import sys
import os
import json
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Groq client ──────────────────────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Models ───────────────────────────────────────────────────────────────────
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "llama-3.3-70b-versatile"

# ── General-purpose MoM system prompt ────────────────────────────────────────
MOM_SYSTEM_PROMPT = """
You are an expert meeting analyst.

Your task is to generate a concise, structured Minutes of Meeting (MoM)
from a raw meeting or call transcript.

PRIMARY OBJECTIVE
• Extract ONLY unique, decision-relevant information
• Merge repeated or rephrased statements into a single insight
• Ignore fillers, greetings, confirmations, and small talk
• Never restate the same fact in different words

EXTRACTION RULES
• Do NOT infer, assume, or guess missing information
• If a field is not explicitly stated, return null
• Each field must contain information not present in any other field
• Prefer short, factual phrases over long sentences

NORMALIZATION RULES
• Convert vague statements into neutral summaries
  Example: "maybe sometime later" → "Undecided timeline"
• Do NOT add explanations or commentary

OUTPUT REQUIREMENTS
• Return ONLY valid JSON
• Must strictly match the schema below
• No markdown, no explanations, no extra text

SCHEMA
{
  "meeting_title": string | null,
  "participants": [string] | null,
  "date": string | null,
  "duration": string | null,
  "summary": string,
  "key_discussion_points": [string],
  "decisions_made": [string],
  "action_items": [
    {
      "task": string,
      "assignee": string | null,
      "deadline": string | null
    }
  ],
  "follow_up": string | null,
  "sentiment": "Positive" | "Neutral" | "Negative" | "Mixed"
}
"""


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file using Groq Whisper (multilingual)."""
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), audio_file.read()),
            model=STT_MODEL,
            response_format="verbose_json",
        )
    return transcription.text


def generate_mom(transcript: str) -> dict:
    """Send the transcript to Groq LLM and extract a structured JSON MoM."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": MOM_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        temperature=0.2,
        max_tokens=700,
    )

    raw = response.choices[0].message.content.strip()

    # Extract the JSON object from the response
    json_start = raw.find("{")
    json_end = raw.rfind("}") + 1
    return json.loads(raw[json_start:json_end])


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python generate_mom_from_audio.py <path_to_audio_file>")
        print("  python generate_mom_from_audio.py --dir <directory_path>")
        sys.exit(1)

    # ── Directory mode: list files and let user pick one ─────────────────
    if sys.argv[1] == "--dir":
        dir_path = sys.argv[2] if len(sys.argv) > 2 else "../../call-recordings"
        if not os.path.isdir(dir_path):
            print(f"Error: Directory not found — {dir_path}")
            sys.exit(1)

        audio_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
        files = sorted([
            f for f in os.listdir(dir_path)
            if os.path.splitext(f)[1].lower() in audio_extensions
        ])

        if not files:
            print("No audio files found in directory.")
            sys.exit(1)

        print(f"\nAudio files in {dir_path}:\n")
        for i, f in enumerate(files, 1):
            print(f"  {i}. {f}")

        choice = input("\nEnter file number: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                audio_path = os.path.join(dir_path, files[idx])
            else:
                print("Invalid selection.")
                sys.exit(1)
        except ValueError:
            print("Invalid input.")
            sys.exit(1)
    else:
        audio_path = sys.argv[1]

    if not os.path.isfile(audio_path):
        print(f"Error: File not found — {audio_path}")
        sys.exit(1)

    # ── Step 1: Transcribe ───────────────────────────────────────────────
    print(f"Transcribing: {audio_path}")
    transcript = transcribe_audio(audio_path)
    print(f"\n── Transcript ─────────────────────────────────────────")
    print(transcript)

    # ── Step 2: Generate MoM ─────────────────────────────────────────────
    print(f"\nGenerating MoM...")
    mom = generate_mom(transcript)

    # Add metadata
    mom["_meta"] = {
        "source_audio": os.path.basename(audio_path),
        "generated_at": datetime.now().isoformat(),
        "stt_model": STT_MODEL,
        "llm_model": LLM_MODEL,
    }

    # ── Step 3: Save & print ─────────────────────────────────────────────
    mom_json = json.dumps(mom, indent=2, ensure_ascii=False)

    # Save JSON to audio-mom directory
    mom_dir = os.path.join(os.path.dirname(__file__), "..", "..", "audio-mom")
    os.makedirs(mom_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(mom_dir, f"{base}_mom.json")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mom_json)

    print(f"\n── MoM (JSON) ────")
    print(mom_json)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
