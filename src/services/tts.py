import socket
import struct
import subprocess
import time
from multiprocessing import Queue

PIPER_BINARY = "piper"
MODEL_PATH = "../piper-models/en_US-ryan-low.onnx"
TTS_PORT = 9998

def create_rtp_packet(seq, ts, ssrc, payload):
    header = struct.pack("!BBHLL", 0x80, 0, seq, ts, ssrc)
    return header + payload

def run_piper(out_queue: Queue,user_speeaking_event,ai_speaking_event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", TTS_PORT))
    
    print(f"TTS waiting for Asterisk on {TTS_PORT}...")

    _, addr = sock.recvfrom(1024)
    target_addr = addr
    print(f"TTS Connected to Asterisk at {target_addr}")

    seq, ts, ssrc = 0, 0, 12345
    last_time=time.time()
    while True:
        text = out_queue.get()
        if text is None: 
            break
        elapsed = time.time() - last_time
        samples_elapsed = int(elapsed * 8000)
        ts = (ts + samples_elapsed) & 0xFFFFFFFF
        if user_speeaking_event.is_set():
            continue
        print("TTS:", text)
        # Pipeline: Piper -> FFmpeg (convert to 8k u-law) -> Python Stdout
        piper_cmd = [
            PIPER_BINARY, "--model", 
            MODEL_PATH, 
            "--output_file", 
            "-", 
            "--output_raw"
            ]
        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "s16le",
            "-ar", "16000",     
            "-ac", "1",
            "-i", "-",
            "-af", "volume=3.0", 
            "-f", "mulaw",
            "-ar", "8000",
            "-ac", "1",
            "-"
        ]
        try:
            p_piper = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            p_ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=p_piper.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
            p_piper.stdin.write(text.encode("utf-8") + b"\n")
            p_piper.stdin.close()
            
            while True:
                if user_speeaking_event.is_set():
                    print("Barge in : stopped rtp stream")

                    ai_speaking_event.clear()

                    p_piper.terminate()
                    p_ffmpeg.terminate()

                    p_piper.wait()
                    p_ffmpeg.wait()

                    while not out_queue.empty():
                        try:
                            out_queue.get_nowait()
                        except:
                            break
                    break

                # 160 bytes = 20ms of audio at 8kHz
                chunk = p_ffmpeg.stdout.read(160)
                if not chunk: 
                    break
                
                chunk = chunk.ljust(160, b'\x00')

                packet = create_rtp_packet(seq, ts, ssrc, chunk)
                sock.sendto(packet, target_addr)
                last_time=time.time()
                seq = (seq + 1) & 0xFFFF
                ts = (ts + 160) & 0xFFFFFFFF
                time.sleep(0.02) # Timing is critical for audio stability

            p_ffmpeg.wait()
        except Exception as e:
            print(f"TTS Error: {e}")
        ai_speaking_event.clear()