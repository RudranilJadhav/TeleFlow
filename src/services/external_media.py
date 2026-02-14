import threading
import numpy as np
import ffmpeg
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad
from multiprocessing import Queue
from scipy.signal import butter, lfilter
import noisereduce as nr

# =========================
# AUDIO DSP UTILITIES
# =========================

def bandpass(x, fs, low=300, high=3400, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, x)

def reduce_noise(x, fs):
    return nr.reduce_noise(
        y=x,
        sr=fs,
        stationary=True,
        prop_decrease=0.8
    )

def suppress_cough(x):
    rms = np.sqrt(np.mean(x**2))
    if rms > 0.08:          # tuned for 16 kHz speech
        x *= 0.3            # attenuate transient
    return x

def limiter(x):
    return np.tanh(x * 2.0)

def preprocess_audio(x, fs=16000):
    x = x.astype(np.float32)

    # DC removal
    x -= np.mean(x)

    # Speech band
    x = bandpass(x, fs)

    # Noise suppression (only if long enough)
    if len(x) > fs * 0.4:
        x = reduce_noise(x, fs)

    # Cough / transient suppression
    x = suppress_cough(x)

    # Soft limiter
    x = limiter(x)

    return x


# =========================
# FFMPEG STDERR LOGGER
# =========================

def read_ffmpeg_stderr(process):
    for line in iter(process.stderr.readline, b""):
        print("ffmpeg:", line.decode(errors="ignore").strip())


# =========================
# MAIN STREAM → VAD → WHISPER
# =========================

def stream_to_whisper(
    text_queue: Queue,
    out_queue: Queue,
    user_speaking_event,
    ai_speaking_event
):
    # ---- Load models
    vad_model = load_silero_vad().to("cpu").eval()

    whisper_model = WhisperModel(
        "small.en",
        device="cpu",
        compute_type="int8"
    )

    # ---- Start FFmpeg RTP decoder
    process = (
        ffmpeg
        .input(
            "../utils/pcmu.sdp",
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

    # ---- Audio constants
    SAMPLE_RATE = 16000
    FRAME_SAMPLES = 320        # 20 ms
    FRAME_BYTES = FRAME_SAMPLES * 2
    VAD_SAMPLES = 512

    vad_buffer = np.zeros(0, dtype=np.float32)
    speech_buffer = np.zeros(0, dtype=np.float32)
    silence_frames = 0

    print("Waiting for speech ...")

    # =========================
    # STREAM LOOP
    # =========================
    while True:
        raw = process.stdout.read(FRAME_BYTES)
        if len(raw) < FRAME_BYTES:
            continue

        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        vad_buffer = np.concatenate([vad_buffer, pcm])

        # ---- Process VAD frames
        while len(vad_buffer) >= VAD_SAMPLES:
            chunk = vad_buffer[:VAD_SAMPLES]
            vad_buffer = vad_buffer[VAD_SAMPLES:]

            with torch.no_grad():
                prob = vad_model(
                    torch.from_numpy(chunk),
                    SAMPLE_RATE
                ).item()

            rms = np.sqrt(np.mean(chunk**2))

            # =========================
            # SPEECH DETECTED
            # =========================
            if prob > 0.7 and rms < 0.2:
                # ---- Barge-in handling
                if ai_speaking_event.is_set():
                    print("Barge-in detected")
                    ai_speaking_event.clear()
                    print("LLM interrupted")

                    # Flush TTS queue
                    while not out_queue.empty():
                        try:
                            out_queue.get_nowait()
                        except:
                            break
                    break

                if not user_speaking_event.is_set():
                    print("User started speaking")

                user_speaking_event.set()
                speech_buffer = np.concatenate([speech_buffer, chunk])
                silence_frames = 0

            # =========================
            # SILENCE
            # =========================
            else:
                silence_frames += 1

            # =========================
            # END OF UTTERANCE
            # =========================
            if silence_frames >= 15:
                user_speaking_event.clear()

                if len(speech_buffer) >= int(SAMPLE_RATE * 0.6):
                    clean_audio = preprocess_audio(
                        speech_buffer,
                        SAMPLE_RATE
                    )

                    segments, _ = whisper_model.transcribe(
                        clean_audio,
                        beam_size=3,
                        vad_filter=False   # Silero already did VAD
                    )

                    for seg in segments:
                        text = seg.text.strip()
                        if text:
                            text_queue.put(text)

                speech_buffer = np.zeros(0, dtype=np.float32)
                silence_frames = 0
