import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from sarvamai import SarvamAI
from groq import Groq

load_dotenv()

# ── Directories ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRANSCRIPTS_DIR = PROJECT_ROOT / "audio-transcripts"
MOM_DIR = PROJECT_ROOT / "audio-mom"

TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
MOM_DIR.mkdir(parents=True, exist_ok=True)

# ── Sarvam AI: Speech-to-Text with Diarization ──────────────

def transcribe(sarvam_key, audio_paths):
    """Upload audio files to Sarvam AI batch STT and download diarized transcripts."""

    client = SarvamAI(api_subscription_key=sarvam_key)

    job = client.speech_to_text_job.create_job(
        model="saaras:v3",
        mode="translate",
        language_code="unknown",
        with_diarization=True,
        num_speakers=2,
    )

    print(f"Uploading {len(audio_paths)} file(s) to Sarvam AI...")
    job.upload_files(file_paths=audio_paths)
    job.start()

    print("Transcribing (this may take a few minutes)...")
    job.wait_until_complete()

    results = job.get_file_results()

    for f in results["successful"]:
        print(f"  ✓ {f['file_name']}")
    for f in results.get("failed", []):
        print(f"  ✗ {f['file_name']} — {f['error_message']}")

    if results["successful"]:
        job.download_outputs(output_dir=str(TRANSCRIPTS_DIR))
        print(f"Transcripts saved to {TRANSCRIPTS_DIR}\n")

# ── Extract diarized text from transcript JSON ───────────────

def build_speaker_text(transcript_json):
    """Convert a Sarvam transcript JSON into labelled speaker text."""

    data = transcript_json

    # Prefer diarized output
    diarized = data.get("diarized_transcript", {})
    entries = diarized.get("entries", [])

    if entries:
        lines = []
        for e in entries:
            speaker = e.get("speaker_id", "?")
            text = e.get("transcript", "").strip()
            if text:
                lines.append(f"Speaker {speaker}: {text}")
        return "\n".join(lines)

    # Fallback to flat transcript
    return data.get("transcript", "")

# ── Groq: Generate MoM ──────────────────────────────────────

MOM_SYSTEM_PROMPT = """\
You are an expert meeting analyst.

You will receive a diarized call transcript with speaker labels.
Produce a detailed, structured Minutes of Meeting (MoM).

Rules:
• Extract ONLY information explicitly stated in the transcript.
• Do NOT infer, assume, or fabricate any detail.
• If a field has no data, use null or an empty list.
• Return ONLY valid JSON matching the schema below — no markdown fences, no commentary.

JSON Schema:
{
  "meeting_title": "string",
  "participants": ["string"],
  "summary": "string — 3-5 sentence overview of the entire conversation",
  "key_discussion_points": [
    {
      "topic": "string",
      "details": "string — what was said about this topic",
      "raised_by": "string | null"
    }
  ],
  "decisions_made": ["string"],
  "action_items": [
    {
      "task": "string",
      "assignee": "string | null",
      "deadline": "string | null"
    }
  ],
  "follow_up": "string | null",
  "sentiment": "Positive | Neutral | Negative | Mixed"
}
"""

def generate_mom(groq_key):
    """Read every transcript JSON and produce a MoM JSON via Groq."""

    client = Groq(api_key=groq_key)
    json_files = sorted(TRANSCRIPTS_DIR.glob("*.json"))

    if not json_files:
        print("No transcript JSONs found — nothing to summarise.")
        return

    print(f"Generating MoM for {len(json_files)} transcript(s)...\n")

    for fp in json_files:
        out_path = MOM_DIR / (fp.stem + "_mom.json")
        if out_path.exists():
            print(f"  ⏭  {fp.name} (MoM already exists)")
            continue

        data = json.loads(fp.read_text(encoding="utf-8"))
        speaker_text = build_speaker_text(data)

        if not speaker_text.strip():
            print(f"  ⏭  {fp.name} (empty transcript)")
            continue

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            messages=[
                {"role": "system", "content": MOM_SYSTEM_PROMPT},
                {"role": "user",   "content": speaker_text},
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Parse the JSON from the LLM response
        start = raw.find("{")
        end = raw.rfind("}") + 1

        if start < 0 or end <= start:
            print(f"  ✗ {fp.name} — could not parse LLM JSON")
            continue

        mom = json.loads(raw[start:end])

        # Attach metadata
        mom["_meta"] = {
            "source_file": fp.stem,
            "stt": "sarvam_saaras_v3",
            "llm": "groq_llama-3.3-70b",
        }

        out_path.write_text(json.dumps(mom, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  ✓ {out_path.name}")

    print("\nDone!")

# ── Pipeline ─────────────────────────────────────────────────

def run(audio_paths, sarvam_key, groq_key):
    transcribe(sarvam_key, audio_paths)
    generate_mom(groq_key)

# ── CLI ──────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python generate_mom_from_audio.py <audio_file>")
        print("  python generate_mom_from_audio.py --dir <directory>")
        sys.exit(1)

    sarvam_key = os.getenv("SARVAM_API_KEY")
    groq_key   = os.getenv("GROQ_API_KEY")

    if not sarvam_key:
        sys.exit("Error: SARVAM_API_KEY not set in .env")
    if not groq_key:
        sys.exit("Error: GROQ_API_KEY not set in .env")

    audio_paths = []

    if sys.argv[1] == "--dir":
        dir_path = sys.argv[2] if len(sys.argv) > 2 else str(PROJECT_ROOT / "call-recordings")

        if not os.path.isdir(dir_path):
            sys.exit(f"Error: Directory not found — {dir_path}")

        exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
        files = sorted(f for f in os.listdir(dir_path)
                        if os.path.splitext(f)[1].lower() in exts)

        if not files:
            sys.exit("No audio files found in directory.")

        print(f"\nFiles in {dir_path}:\n")
        for i, name in enumerate(files, 1):
            print(f"  {i}. {name}")
        print("\n  a. Process ALL\n")

        choice = input("Pick a number or 'a': ").strip().lower()

        if choice == "a":
            audio_paths = [os.path.abspath(os.path.join(dir_path, f)) for f in files]
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    audio_paths = [os.path.abspath(os.path.join(dir_path, files[idx]))]
                else:
                    sys.exit("Invalid selection.")
            except ValueError:
                sys.exit("Invalid input.")
    else:
        fp = sys.argv[1]
        if not os.path.isfile(fp):
            sys.exit(f"Error: File not found — {fp}")
        audio_paths = [os.path.abspath(fp)]

    run(audio_paths, sarvam_key, groq_key)


if __name__ == "__main__":
    main()