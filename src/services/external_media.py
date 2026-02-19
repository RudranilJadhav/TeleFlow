import threading
import numpy as np
import ffmpeg
from faster_whisper import WhisperModel
from multiprocessing import Queue
from scipy.signal import butter, lfilter
import noisereduce as nr
import torch
from vad_with_bargein import VADWithBargeIn, BargeInConfig


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

def preprocess_audio(x, fs=16000):
    x = x.astype(np.float32)
    x -= np.mean(x)
    x = bandpass(x, fs)
    
    if len(x) > fs * 0.4:
        x = reduce_noise(x, fs)
    
    return x

# FFMPEG STDERR LOGGER
def read_ffmpeg_stderr(process):
    for line in iter(process.stderr.readline, b""):
        print("ffmpeg:", line.decode(errors="ignore").strip())


# VAD
def stream_to_whisper(
    text_queue: Queue,
    out_queue: Queue,
    user_speaking_event,
    ai_speaking_event
):
    # Configure barge-in detection
    config = BargeInConfig()
    config.vad_threshold = 0.3
    config.speech_threshold = 0.4
    config.min_speech_for_bargein = 200
    config.min_utterance_ms = 250
    config.silence_timeout_ms = 250
    config.barge_in_cooldown_ms = 800
    config.hangover_ms = 200
    config.vad_smoothing_window = 3
    config.noise_floor_alpha = 0.995

    # Create VAD with barge-in
    vad = VADWithBargeIn(config)

    
    # Load Whisper model
    whisper_model = WhisperModel(
        "tiny.en",
        device="cuda",
        compute_type="int8"
    )   
    
    # Start FFmpeg RTP decoder
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
    
    # Audio constants
    SAMPLE_RATE = 16000
    FRAME_SAMPLES = 320  # 20ms frames for reading
    FRAME_BYTES = FRAME_SAMPLES * 2
    VAD_FRAME_SAMPLES = int(SAMPLE_RATE * 32 / 1000)  # 32ms for VAD
    
    # Buffers
    audio_buffer = np.zeros(0, dtype=np.float32)
    
    print("Waiting for speech ...")
    
    while True:
        # Read audio from FFmpeg
        raw = process.stdout.read(FRAME_BYTES)
        if len(raw) < FRAME_BYTES:
            continue
        
        # Convert to float32
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        audio_buffer = np.concatenate([audio_buffer, pcm])
        
        # Process in VAD-sized chunks
        while len(audio_buffer) >= VAD_FRAME_SAMPLES:
            chunk = audio_buffer[:VAD_FRAME_SAMPLES]
            audio_buffer = audio_buffer[VAD_FRAME_SAMPLES:]
            
            # Process through VAD with barge-in
            result = vad.process_frame(chunk)
            
            # Handle barge-in
            if result['should_barge_in'] and ai_speaking_event.is_set():
                print("🚨 BARGE-IN DETECTED - Interrupting")
                ai_speaking_event.clear()
                user_speaking_event.set()
                
                # Flush TTS queue
                while not out_queue.empty():
                    try:
                        out_queue.get_nowait()
                    except:
                        break
            
            # Update user speaking state
            if result['is_speech'] and not ai_speaking_event.is_set():
                user_speaking_event.set()
            elif not result['is_speech'] and not ai_speaking_event.is_set():
                # Only clear if we're sure they're done
                if not result['utterance_complete']:
                    pass  # Keep speaking state if utterance not complete
                else:
                    user_speaking_event.clear()
            
            # Handle complete utterance
            if result['utterance_complete'] and result['utterance_audio'] is not None:
                audio = result['utterance_audio']
                
                # Check minimum length
                if len(audio) >= int(SAMPLE_RATE * config.min_utterance_ms / 1000):
                    # Preprocess audio
                    clean_audio = preprocess_audio(audio, SAMPLE_RATE)
                    
                    # Transcribe
                    segments, _ = whisper_model.transcribe(
                        clean_audio,
                        beam_size=3,
                        vad_filter=False
                    )
                    
                    for seg in segments:
                        text = seg.text.strip()
                        if text:
                            # Check if it's just a backchannel
                            if vad.is_backchannel(text):
                                print(f"Backchannel ignored: '{text}'")
                                continue
                            
                            print(f"User: {text}")
                            text_queue.put(text)
                else:
                    print(f"Utterance too short ({len(audio)/SAMPLE_RATE:.2f}s), ignoring")