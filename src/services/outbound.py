import threading
import time
import json
import requests
import websocket
from multiprocessing import Process, Queue, Event
import os

# Import capabilities from your existing files
from brain import run_llm
from external_media import stream_to_whisper
from tts import run_piper
# We import specific functions/vars from events to reuse the setup logic
from events import setup_call, calls, calls_lock, ari_request, APP_NAME, WS_URL, USER, PASSWORD, setup_semaphore

# --- Configuration ---
TARGET_NUMBER = "PJSIP/6001"  # The Zoiper user to call
BOT_CALLER_ID = "Stark Bot <8888>"

def initiate_outbound_call(target_endpoint, caller_id):
    """Triggers the call via ARI"""
    url = f"http://localhost:8088/ari/channels"
    params = {
        "endpoint": target_endpoint,
        "extension": "1000",
        "context": "default",
        "priority": 1,
        "app": APP_NAME,        # Must match the app name in the websocket listener below
        "appArgs": "outbound",
        "callerId": caller_id,
        "timeout": 30
    }
    try:
        print(f"📞 Dialing {target_endpoint}...")
        requests.post(url, auth=(USER, PASSWORD), json=params)
    except Exception as e:
        print(f"Error triggering call: {e}")

def outbound_event_listener(transcript_queue, text_queue):
    """
    A custom event listener specifically for this outbound script.
    It reuses 'setup_call' from events.py but adds the 'Hello' trigger.
    """
    conversation = []
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

            print(f"🚀 Call Connected: {channel_id}")
            
            with lock:
                call_active = True
            
            # A. Run the standard setup (Bridge, ASR, TTS)
            # We run this in a thread to not block the websocket
            def setup_and_greet():
                # Reuse the existing setup logic from events.py
                setup_call(channel_id)
                
                # B. THE MAGIC FIX: Manually trigger the bot
                print("🤖 Triggering Artificial Greeting...")
                time.sleep(1) # Small buffer to ensure audio path is ready
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
            
            # (Optional) You can add the MoM generation logic here if you want it
            if call_active:
                print("Call ended. Shutting down script in 3 seconds...")
                # We stop the script shortly after the call ends
                threading.Timer(3.0, ws.close).start()

    # Connect to Asterisk
    ws_url = f"{WS_URL}?api_key={USER}:{PASSWORD}&app={APP_NAME}"
    ws = websocket.WebSocketApp(ws_url, on_message=on_message)
    print("⚡ Listening for Stasis Events...")
    ws.run_forever()

if __name__ == "__main__":
    print("--- STARTING INDEPENDENT OUTBOUND AGENT ---")
    
    # 1. Initialize Queues
    text_queue = Queue()
    out_queue = Queue()
    transcript_queue = Queue()
    text_queue.put("Hello")
    
    # 2. Shared Events
    user_speaking_event = Event()
    ai_speaking_event = Event()

    # 3. Start AI Processes (Same as main.py)
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

    # 4. Start the Event Listener (In a separate process or thread)
    # We run it as a Process to keep it clean, but Thread is easier for 'initiate_call' timing
    t_events = threading.Thread(
        target=outbound_event_listener,
        args=(transcript_queue, text_queue)
    )
    t_events.start()

    # 5. Wait a moment for everything to spin up, then Dial
    time.sleep(2)
    initiate_outbound_call(TARGET_NUMBER, BOT_CALLER_ID)

    # 6. Keep main thread alive until events thread finishes (when call ends)
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