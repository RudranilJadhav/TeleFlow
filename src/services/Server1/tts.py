import socket
import struct
import subprocess
import time
from multiprocessing import Queue

PIPER_BINARY = "piper"
MODEL_PATH = "en_US-ryan-low.onnx"
TTS_PORT = 9998

def create_rtp_packet(seq, ts, ssrc, payload):
    # Header: Version 2, Payload Type 0 (PCMU/u-law)
    header = struct.pack("!BBHLL", 0x80, 0, seq, ts, ssrc)
    return header + payload

def run_piper(out_queue: Queue):
    # Setup UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", TTS_PORT))
    
    print(f"TTS waiting for Asterisk on {TTS_PORT}...")
    # Wait for Asterisk to send a packet so we know its IP/Port
    _, addr = sock.recvfrom(1024)
    target_addr = addr
    print(f"TTS Connected to Asterisk at {target_addr}")

    seq, ts, ssrc = 0, 0, 12345

    while True:
        text = out_queue.get()
        if text is None: break
        print("TTS:", text)
        # Pipeline: Piper -> FFmpeg (convert to 8k u-law) -> Python Stdout
        piper_cmd = [PIPER_BINARY, "--model", MODEL_PATH, "--output_file", "-", "--output_raw"]
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
                # 160 bytes = 20ms of audio at 8kHz
                chunk = p_ffmpeg.stdout.read(160)
                if not chunk: break
                
                chunk = chunk.ljust(160, b'\x00')

                packet = create_rtp_packet(seq, ts, ssrc, chunk)
                sock.sendto(packet, target_addr)

                seq = (seq + 1) & 0xFFFF
                ts = (ts + 160) & 0xFFFFFFFF
                time.sleep(0.02) # Timing is critical for audio stability

            p_ffmpeg.wait()
        except Exception as e:
            print(f"TTS Error: {e}")