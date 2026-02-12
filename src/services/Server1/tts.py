# import sounddevice as sd
# from piper import PiperVoice

# def run_piper(out_queue):
#     voice = PiperVoice.load(
#         "en_US-ryan-high.onnx",
#         config_path="en_US-ryan-high.onnx.json"
#     )

#     while True:
#         text = out_queue.get()
#         if text is None:
#             break

#         for chunk in voice.synthesize(text):
#             # chunk.samples is already a NumPy int16 array
#             sd.play(chunk.samples, samplerate=chunk.sample_rate)
#             sd.wait()


import subprocess
from piper import PiperVoice

def run_piper(out_queue):
    voice = PiperVoice.load(
        "en_US-ryan-high.onnx",
        config_path="en_US-ryan-high.onnx.json"
    )

    aplay = None

    try:
        while True:
            text = out_queue.get()
            if text is None:
                break

            for chunk in voice.synthesize(text):
                audio_bytes = chunk.audio_int16_bytes
                sample_rate = chunk.sample_rate

                if aplay is None:
                    aplay = subprocess.Popen(
                        [
                            "aplay",
                            "-q",
                            "-f", "S16_LE",
                            "-c", "1",
                            "-r", str(sample_rate)
                        ],
                        stdin=subprocess.PIPE
                    )

                aplay.stdin.write(audio_bytes)

    finally:
        if aplay:
            try:
                aplay.stdin.close()
                aplay.wait()
            except Exception:
                pass
