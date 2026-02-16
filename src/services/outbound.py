import threading
import time
import json
import requests
import websocket
from multiprocessing import Process, Queue, Event


from brain import run_llm
from external_media import stream_to_whisper
from tts import run_piper
from events import setup_call, calls, calls_lock, ari_request, APP_NAME, WS_URL, USER, PASSWORD


TARGET_NUMBER = "PJSIP/6001"
BOT_CALLER_ID = "Jarvis <8888>"

def initiate_outbound_call(target_endpoint, caller_id):
    url = f"http://localhost:8088/ari/channels"
    params = {
        "endpoint": target_endpoint,
        "extension": "1000",
        "context": "default",
        "priority": 1,
        "app": APP_NAME,
        "appArgs": "outbound",
        "callerId": caller_id,
        "timeout": 30
    }
    try:
        print(f"Dialing {target_endpoint}...")
        requests.post(url, auth=(USER, PASSWORD), json=params)
    except Exception as e:
        print(f"Error triggering call: {e}")

def outbound_event_listener(text_queue,transcript_queue):
    call_active = False
    lock = threading.Lock()

    def on_message(ws, message):
        nonlocal call_active
        event = json.loads(message)
        
        # 1. Call Connected
        if event.get("type") == "StasisStart":
            channel_id = event["channel"]["id"]
            c_name = event.get("channel", {}).get("name", "")
            
            # Ignore internal channels
            if c_name.startswith("UnicastRTP") or c_name.startswith("Snoop"):
                return

            print(f"Call Connected: {channel_id}")
            
            with lock:
                call_active = True
            
            # A. Run the standard setup (Bridge, ASR, TTS)
            def setup_and_greet():
                # Reuse the existing setup logic from events.py
                setup_call(channel_id)
                
                print("Triggering Artificial Greeting...")
                time.sleep(1) # Small buffer to ensure everything is ready
                text_queue.put("Hello") 

            threading.Thread(target=setup_and_greet).start()

        # 2. Call Ended
        elif event.get("type") == "StasisEnd":
            cid = event.get("channel", {}).get("id")
            
            # Cleanup bridges using the existing logic
            with calls_lock:
                data = calls.pop(cid, None)
                if data:
                    print(f"Cleaning up call: {cid}")
                    ari_request('DELETE', f"bridges/{data['main_bridge']}")
                    ari_request('DELETE', f"bridges/{data['asr_bridge']}")
            
            with lock:
                call_active = False
            print("Call ended. Shutting down script in 3 seconds...")
            threading.Timer(3.0, ws.close).start()

    # Connect to Asterisk
    ws_url = f"{WS_URL}?api_key={USER}:{PASSWORD}&app={APP_NAME}"
    ws = websocket.WebSocketApp(ws_url, on_message=on_message)
    print("Listening for Stasis Events...")
    ws.run_forever()

if __name__ == "__main__":
    print("--- STARTING OUTBOUND CALL---")
    
    text_queue = Queue()
    out_queue = Queue()
    transcript_queue = Queue()
    text_queue.put("Hello")
    
    user_speaking_event = Event()
    ai_speaking_event = Event()

    p_llm = threading.Thread(
        target=run_llm,
        args=(text_queue, out_queue, user_speaking_event, ai_speaking_event, transcript_queue,"Outbound"),
        daemon=True
    )
    p_llm.start()

    p_asr = Process(
        target=stream_to_whisper,
        args=(text_queue, out_queue, user_speaking_event, ai_speaking_event),
        name="ASR"
    )
    
    p_piper = Process(
        target=run_piper,
        args=(out_queue, user_speaking_event, ai_speaking_event),
        name="PIPER"
    )
    
    p_asr.start()
    p_piper.start()

    t_events = threading.Thread(
        target=outbound_event_listener,
        args=(text_queue,)
    )
    t_events.start()

    time.sleep(2)
    initiate_outbound_call(TARGET_NUMBER, BOT_CALLER_ID)

    try:
        t_events.join()
        p_asr.join()
        p_piper.join()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        p_asr.terminate()
        p_piper.terminate()
        print("Done.")