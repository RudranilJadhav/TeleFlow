import threading
import numpy as np
import ffmpeg
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad


vad_model = load_silero_vad().to("cpu")

whisper_model = WhisperModel(
    "small",
    device="cuda",
    compute_type="int8"
)

def read_ffmpeg_stderr(process):
    for line in iter(process.stderr.readline, b""):
        print("ffmpeg:", line.decode(errors="ignore").strip())

def stream_to_whisper():
    process = (
        ffmpeg
        .input(
            "src/services/pcmu.sdp",
            protocol_whitelist="file,udp,rtp",
            fflags="nobuffer",
            flags="low_delay",
            analyzeduration=0,
            probesize=32
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

    threading.Thread(
        target=read_ffmpeg_stderr,
        args=(process,),
        daemon=True
    ).start()

    SAMPLE_RATE = 16000
    FRAME_MS = 20
    FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 320
    FRAME_BYTES = FRAME_SAMPLES * 2

    VAD_WINDOW_MS = 100
    VAD_SAMPLES = 512

    vad_buffer = np.zeros(0, dtype=np.float32)
    speech_buffer = np.zeros(0, dtype=np.float32)

    speech_frames = 0
    silence_frames = 0

    print("Waiting for speech ...")
    while True:
        raw = process.stdout.read(FRAME_BYTES)
        if len(raw) < FRAME_BYTES:
            continue

        pcm16 = np.frombuffer(raw, dtype=np.int16)
        pcmf = pcm16.astype(np.float32) / 32768.0

        vad_buffer = np.concatenate([vad_buffer, pcmf])

        while len(vad_buffer) >= VAD_SAMPLES:
            chunk = vad_buffer[:VAD_SAMPLES]
            vad_buffer = vad_buffer[VAD_SAMPLES:]

            with torch.no_grad():
                prob = vad_model(
                    torch.from_numpy(chunk),
                    SAMPLE_RATE
                ).item()

            if prob > 0.6:
                speech_frames += 1
                silence_frames = 0
                speech_buffer = np.concatenate([speech_buffer, chunk])
            else:
                silence_frames += 1

            if silence_frames >= 5 and len(speech_buffer) > SAMPLE_RATE:

                segments, _ = whisper_model.transcribe(
                    speech_buffer,
                    beam_size=5,
                    vad_filter=False
                )

                for seg in segments:
                    text = seg.text.strip()
                    if text:
                        print(f"[{seg.start:.2f}s → {seg.end:.2f}s] {text}")

                speech_buffer = np.zeros(0, dtype=np.float32)
                speech_frames = 0
                silence_frames = 0
                print("Waiting for speech ...")
if __name__ == "__main__":
    stream_to_whisper()
