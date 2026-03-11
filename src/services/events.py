from multiprocessing import Queue
import os
import websocket
import requests
import json
import threading
from mom_generator import generate_mom, generate_mom_document
from datetime import datetime
LIVE_TRANSCRIPT_FILE = "../../call-logs/live_transcript.txt"

ARI_URL = "http://localhost:8088/ari"
WS_URL = "ws://localhost:8088/ari/events"
USER = "6001"
PASSWORD = "unsecurepassword"
APP_NAME = "stream-app"

calls = {}
calls_lock = threading.Lock()

setup_semaphore = threading.BoundedSemaphore(5)

def ari_request(method, endpoint, data=None):
    url = f"{ARI_URL}/{endpoint}"
    try:
        if method == 'POST':
            r = requests.post(url, auth=(USER, PASSWORD), json=data)
        elif method == 'DELETE':
            r = requests.delete(url, auth=(USER, PASSWORD))
        elif method == 'GET':
            r = requests.get(url, auth=(USER, PASSWORD))
        else: return None
        r.raise_for_status()
        return r.json() if r.text and r.status_code != 204 else {}
    except Exception as e:
       
        print(f"ARI Error [{endpoint}]: {e}")
        return None

def setup_call(channel_id):

    with setup_semaphore:
        print(f"Processing: {channel_id}")

        # 1. Answer and create Main Bridge (Caller + TTS)
        ari_request('POST', f"channels/{channel_id}/answer")
        main_bridge = ari_request('POST', "bridges", {"type": "mixing"})
        
        if not main_bridge: 
            print(f"Failed to create bridge for {channel_id}")
            return
            
        main_bridge_id = main_bridge["id"]
        ari_request('POST', f"bridges/{main_bridge_id}/addChannel", {"channel": channel_id})

        # 2. Add TTS on Port 9998
        tts_ch = ari_request('POST', "channels/externalMedia", {
            "app": APP_NAME, 
            "external_host": "127.0.0.1:9998", 
            "format": "ulaw",
            "direction": "recv" 
        })
        
        if tts_ch:
            ari_request('POST', f"bridges/{main_bridge_id}/addChannel", {"channel": tts_ch["id"]})

        # 3. Create Snoop Channel
        # spy='in': Listen ONLY to the user, not the TTS
        snoop_ch = ari_request('POST', f"channels/{channel_id}/snoop", {
            "app": APP_NAME,
            "spy": "in"
        })

        if not snoop_ch:
             print(f"Snoop failed for {channel_id}")
             return

        # 4. Add ASR on Port 9999
        asr_ch = ari_request('POST', "channels/externalMedia", {
            "app": APP_NAME, 
            "external_host": "127.0.0.1:9999", 
            "format": "ulaw", 
            "direction": "send"
        })

        if asr_ch:
            # Create a dedicated bridge for the Snoop -> ASR connection
            asr_bridge = ari_request('POST', "bridges", {"type": "mixing"})
            ari_request('POST', f"bridges/{asr_bridge['id']}/addChannel", 
                        {"channel": [snoop_ch["id"], asr_ch["id"]]})

            # 5. Start recording on the main bridge (captures both sides)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rec_name = f"recording_{timestamp}"
            ari_request('POST', f"bridges/{main_bridge_id}/record", {
                "name": rec_name,
                "format": "wav",
                "ifExists": "overwrite"
            })

            with calls_lock:
                calls[channel_id] = {
                    "main_bridge": main_bridge_id, 
                    "asr_bridge": asr_bridge['id'],
                    "recording_name": rec_name
                }
            print(f"Active: {channel_id} | Main: {main_bridge_id} | ASR: {asr_bridge['id']} | Rec: {rec_name}")

                
def run(transcript_queue: Queue,text_queue: Queue):
    # Shared state for conversation tracking
    conversation = []
    call_active = False
    lock = threading.Lock()

    # Thread to listen for transcript lines
    def queue_listener():
        nonlocal conversation, call_active
        while True:
            line = transcript_queue.get()
            with lock:
                if call_active:
                    conversation.append(line)
                    with open(LIVE_TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
                        f.write(line + "\n")

    threading.Thread(target=queue_listener, daemon=True).start()

    def on_message(ws, message):
        nonlocal conversation, call_active
        event = json.loads(message)
        if event.get("type") == "StasisStart":
            # Ignore specialized channels
            c_name = event.get("channel", {}).get("name", "")
            if c_name.startswith("UnicastRTP") or c_name.startswith("Snoop"):
                return

            # Reset conversation for new call
            with lock:
                call_active = True
                conversation.clear()

            threading.Thread(target=setup_call, args=(event["channel"]["id"],)).start()

        elif event.get("type") == "StasisEnd":
            cid = event.get("channel", {}).get("id")
            with calls_lock:
                data = calls.pop(cid, None)
                if data:
                    # Save the call recording BEFORE destroying bridges
                    rec_name = data.get('recording_name')
                    if rec_name:
                        try:
                            # Stop the live recording first
                            ari_request('POST', f"recordings/live/{rec_name}/stop")
                            # Download the stored recording file
                            rec_url = f"{ARI_URL}/recordings/stored/{rec_name}/file"
                            r = requests.get(rec_url, auth=(USER, PASSWORD))
                            if r.status_code == 200:
                                rec_dir = "../../call-recordings"
                                os.makedirs(rec_dir, exist_ok=True)
                                filepath = os.path.join(rec_dir, f"{rec_name}.wav")
                                with open(filepath, "wb") as f:
                                    f.write(r.content)
                                print(f"Call recording saved to {filepath}")
                            else:
                                print(f"Recording download failed: {r.status_code}")
                            # Clean up stored recording from Asterisk
                            ari_request('DELETE', f"recordings/stored/{rec_name}")
                        except Exception as e:
                            print(f"Recording save error: {e}")

                    # Now destroy bridges
                    ari_request('DELETE', f"bridges/{data['main_bridge']}")
                    ari_request('DELETE', f"bridges/{data['asr_bridge']}")
                    print(f"Ended: {cid}")

            # Generate MoM if we have conversation
            with lock:
                if call_active and conversation:
                    chat_history = "\n".join(conversation)
                    mom = generate_mom(chat_history)
                    doc = generate_mom_document(mom)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_dir = "../../call-logs"
                    os.makedirs(log_dir, exist_ok=True)
                    filename = os.path.join(log_dir, f"mom_{mom['customer_name']}_{cid}_{timestamp}.txt")
                    with open(filename, "w") as f:
                        f.write(doc)
                    print(f"\n=== MoM saved to {filename} ===\n")
                    open(LIVE_TRANSCRIPT_FILE, "w").close()
                call_active = False

    ws_url = f"{WS_URL}?api_key={USER}:{PASSWORD}&app={APP_NAME}"
    ws = websocket.WebSocketApp(ws_url, on_message=on_message)
    ws.run_forever()
