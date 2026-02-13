import websocket
import requests
import json
import threading
import time

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
        # Simple retry logic could be added here, but the Semaphore usually fixes it.
        print(f"ARI Error [{endpoint}]: {e}")
        return None

def setup_call(channel_id):
    # Acquire lock. If 5 calls are setting up, this thread waits here.
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

        # 2. Add TTS (Static Port 9998)
        # Direction RECV: We receive audio FROM 9998
        tts_ch = ari_request('POST', "channels/externalMedia", {
            "app": APP_NAME, 
            "external_host": "127.0.0.1:9998", 
            "format": "ulaw",
            "direction": "recv" 
        })
        
        if tts_ch:
            ari_request('POST', f"bridges/{main_bridge_id}/addChannel", {"channel": tts_ch["id"]})

        # 3. Create Snoop Channel (The Echo Fix)
        # spy='in': Listen ONLY to the user, not the TTS
        snoop_ch = ari_request('POST', f"channels/{channel_id}/snoop", {
            "app": APP_NAME,
            "spy": "in"
        })

        if not snoop_ch:
             print(f"Snoop failed for {channel_id}")
             return

        # 4. Add ASR (Static Port 9999)
        # Direction SEND: We send audio TO 9999
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

            with calls_lock:
                calls[channel_id] = {
                    "main_bridge": main_bridge_id, 
                    "asr_bridge": asr_bridge['id']
                }
            print(f"Active: {channel_id} | Main: {main_bridge_id} | ASR: {asr_bridge['id']}")

def on_message(ws, message):
    event = json.loads(message)
    if event.get("type") == "StasisStart":
        # Ignore specialized channels (Snoop, Recorder, UnicastRTP) to avoid infinite loops
        c_name = event.get("channel", {}).get("name", "")
        if c_name.startswith("UnicastRTP") or c_name.startswith("Snoop"): 
            return
        
        # Spawn thread for non-blocking processing
        threading.Thread(target=setup_call, args=(event["channel"]["id"],)).start()

    elif event.get("type") == "StasisEnd":
        cid = event.get("channel", {}).get("id")
        with calls_lock:
            data = calls.pop(cid, None)
            if data:
                # Cleanup both bridges
                ari_request('DELETE', f"bridges/{data['main_bridge']}")
                ari_request('DELETE', f"bridges/{data['asr_bridge']}")
                print(f"Ended: {cid}")
                
def run():
    ws_url = f"{WS_URL}?api_key={USER}:{PASSWORD}&app={APP_NAME}"
    ws = websocket.WebSocketApp(ws_url, on_message=on_message)
    ws.run_forever()
