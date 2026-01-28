import socket
import numpy as np
from faster_whisper import WhisperModel
import ffmpeg

# Whisper Setup
model = WhisperModel("tiny", device="cuda", compute_type="int8")

def stream_to_whisper():
    # Use ffmpeg to listen to the UDP port and convert to 16k mono PCM
    # This replaces your socket.recvfrom logic
    process = (
        ffmpeg
        .input('udp://0.0.0.0:9999', format='mp3') # Change format if source isn't MP3
        .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    print("Listening for audio...")

    while True:
        # Read 5 seconds of audio at a time (16000 samples * 2 bytes * 5s)
        in_bytes = process.stdout.read(16000 * 2 * 5)
        if not in_bytes:
            break
            
        # Convert bytes to NumPy array
        audio_data = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0
        
        # Transcribe chunk
        segments, _ = model.transcribe(audio_data, beam_size=5)
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

stream_to_whisper()