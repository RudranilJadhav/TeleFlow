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

def run_piper(out_queue: Queue, user_speaking_event, ai_speaking_event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Add this
    sock.bind(("0.0.0.0", TTS_PORT))
    
    print(f"TTS waiting for Asterisk on {TTS_PORT}...")

    _, addr = sock.recvfrom(1024)
    target_addr = addr
    print(f"TTS Connected to Asterisk at {target_addr}")

    seq, ts, ssrc = 0, 0, 12345
    rtp_ts_step = 160  # 20ms at 8kHz = 160 samples
    last_packet_time = time.time()
    
    while True:
        text = out_queue.get()
        if text is None: 
            break
            
        # Check if user started speaking while we were waiting
        if user_speaking_event.is_set():
            print("User speaking, skipping TTS")
            ai_speaking_event.clear()
            continue
            
        # Update timestamp based on elapsed time since last packet
        current_time = time.time()
        elapsed_ms = (current_time - last_packet_time) * 1000
        samples_elapsed = int(elapsed_ms * 8)  # 8 samples per ms at 8kHz
        ts = (ts + samples_elapsed) & 0xFFFFFFFF
        
        print(f"TTS: {text}")
        
        # Pipeline: Piper -> FFmpeg (convert to 8k u-law)
        piper_cmd = [
            PIPER_BINARY, "--model", MODEL_PATH, 
            "--output_file", "-", "--output_raw"
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
            # Start processes
            p_piper = subprocess.Popen(
                piper_cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.DEVNULL
            )
            p_ffmpeg = subprocess.Popen(
                ffmpeg_cmd, 
                stdin=p_piper.stdout, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.DEVNULL
            )
            
            # Feed text to piper
            p_piper.stdin.write(text.encode("utf-8") + b"\n")
            p_piper.stdin.close()
            
            # Stream audio
            barge_in_detected = False
            while True:
                # Check for barge-in with small timeout to be responsive
                if user_speaking_event.is_set():
                    print("Barge-in detected, stopping TTS stream")
                    barge_in_detected = True
                    ai_speaking_event.clear()
                    
                    # Terminate processes
                    p_piper.terminate()
                    p_ffmpeg.terminate()
                    
                    # Wait for them to finish
                    try:
                        p_piper.wait(timeout=1)
                        p_ffmpeg.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        p_piper.kill()
                        p_ffmpeg.kill()
                    
                    # Clear queue
                    while not out_queue.empty():
                        try:
                            out_queue.get_nowait()
                        except:
                            break
                    break
                
                # Read audio chunk (160 bytes = 20ms)
                chunk = p_ffmpeg.stdout.read(160)
                if not chunk:
                    break
                
                # Ensure chunk is exactly 160 bytes
                if len(chunk) < 160:
                    chunk = chunk.ljust(160, b'\x00')
                
                # Send RTP packet
                packet = create_rtp_packet(seq, ts, ssrc, chunk)
                sock.sendto(packet, target_addr)
                
                # Update timestamps
                last_packet_time = current_time
                seq = (seq + 1) & 0xFFFF
                ts = (ts + rtp_ts_step) & 0xFFFFFFFF
                
                # Small sleep to maintain real-time
                time.sleep(0.018)
            
            # If we finished without barge-in, wait for processes
            if not barge_in_detected:
                p_ffmpeg.wait()
                
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            # Ensure event is cleared even if error occurs
            ai_speaking_event.clear()