import threading
import numpy as np
import ffmpeg
from faster_whisper import WhisperModel

model = WhisperModel(
    "small",
    device="cuda",
    compute_type="int8"
)

# =========================
# ffmpeg stderr reader
# =========================
def read_ffmpeg_stderr(process):
    for line in iter(process.stderr.readline, b""):
        print("ffmpeg:", line.decode(errors="ignore").strip())

# =========================
# Main streaming function
# =========================
def stream_to_whisper():
    print("Starting ffmpeg RTP listener...")

    process = (
        ffmpeg
        .input(
            "src/services/pcmu.sdp",
            protocol_whitelist="file,udp,rtp",
            fflags="nobuffer",
            flags="low_delay"
        )
        .output(
            "-",
            format="s16le",
            acodec="pcm_s16le",
            ac=1,
            ar="16000"
        )
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    print("ffmpeg started, pid:", process.pid)

    # Start stderr reader
    threading.Thread(
        target=read_ffmpeg_stderr,
        args=(process,),
        daemon=True
    ).start()

    SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2
    CHUNK_SECONDS = 2

    TARGET_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_SECONDS
    buffer = bytearray()

    print("🎧 Listening for audio...")

    while True:
        data = process.stdout.read(4096)

        if not data:
            continue

        # Debug heartbeat (comment later)
        print("audio bytes:", len(data))

        buffer.extend(data)

        if len(buffer) < TARGET_BYTES:
            continue

        # Take exact chunk
        chunk = buffer[:TARGET_BYTES]
        del buffer[:TARGET_BYTES]

        audio = (
            np.frombuffer(chunk, np.int16)
            .astype(np.float32) / 32768.0
        )

        segments, info = model.transcribe(
            audio,
            beam_size=5,
            vad_filter=False
        )

        for seg in segments:
            text = seg.text.strip()
            if text:
                print(f"[{seg.start:.2f}s → {seg.end:.2f}s] {text}")

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    stream_to_whisper()
