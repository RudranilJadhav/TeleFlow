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
EXTERNAL_MEDIA_HOST = "127.0.0.1:9999"

calls = {}
calls_lock = threading.Lock()

def ari_request(method, endpoint, data=None):
    """Helper for ARI requests with error handling."""
    url = f"{ARI_URL}/{endpoint}"
    try:
        if method == 'POST':
            r = requests.post(url, auth=(USER, PASSWORD), json=data)
        elif method == 'DELETE':
            r = requests.delete(url, auth=(USER, PASSWORD))
        elif method == 'GET':
            r = requests.get(url, auth=(USER, PASSWORD))
        else:
            return None

        r.raise_for_status()
        if r.status_code == 204 or not r.text:
            return {}
        return r.json()
    except requests.exceptions.HTTPError as err:
        if err.response.status_code != 404:
            print(f"ARI Error [{endpoint}]: {err}")
        return None
    except Exception as e:
        print(f"Request Error: {e}")
        return None

def cleanup_all():
    """Aggressive cleanup for development/testing."""
    print("Cleaning up old bridges...")
    bridges = ari_request('GET', 'bridges')
    if bridges:
        for b in bridges:
            if b.get("bridge_type") == "mixing":
                ari_request('DELETE', f"bridges/{b['id']}")

def setup_call(channel_id):
    """Run in a thread to avoid blocking WebSocket pings."""
    print(f"Setting up call for: {channel_id}")
    ari_request('POST', f"channels/{channel_id}/answer")
    bridge = ari_request('POST', "bridges", {"type": "mixing"})
    if not bridge:
        print(f"Failed to create bridge for {channel_id}. Hanging up.")
        ari_request('DELETE', f"channels/{channel_id}")
        return
    bridge_id = bridge["id"]
    ari_request('POST', f"bridges/{bridge_id}/addChannel", {"channel": channel_id})
    external = ari_request('POST', "channels/externalMedia", {
        "app": APP_NAME,
        "external_host": EXTERNAL_MEDIA_HOST,
        "format": "ulaw",
        "direction": "both"
    })

    if not external:
        print("Failed to create external media. Cleaning up.")
        ari_request('DELETE', f"bridges/{bridge_id}")
        ari_request('DELETE', f"channels/{channel_id}")
        return

    media_id = external["id"]
    ari_request('POST', f"bridges/{bridge_id}/addChannel", {"channel": media_id})
    with calls_lock:
        calls[channel_id] = {
            "bridge": bridge_id,
            "media": media_id
        }
    
    print(f"Streaming Active: {channel_id} <-> {media_id} ({EXTERNAL_MEDIA_HOST})")

def on_message(ws, message):
    try:
        event = json.loads(message)
    except json.JSONDecodeError:
        return

    etype = event.get("type")
    channel = event.get("channel", {})
    channel_id = channel.get("id")
    channel_driver = channel.get("name", "").split("/")[0]
    if etype == "StasisStart":
        if channel_driver == "UnicastRTP":
            return
        with calls_lock:
            if channel_id in calls:
                return
        print(f"Call Received: {channel_id} ({channel.get('name')})")
        t = threading.Thread(target=setup_call, args=(channel_id,))
        t.start()
    elif etype == "StasisEnd":
        call_data = None
        with calls_lock:
            call_data = calls.pop(channel_id, None)

        if call_data:
            print(f"Call Ended: {channel_id}. Cleaning up resources.")
            if call_data.get('media'):
                ari_request('DELETE', f"channels/{call_data['media']}")
            if call_data.get('bridge'):
                ari_request('DELETE', f"bridges/{call_data['bridge']}")

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed.")

def on_open(ws):
    print(f"Connected to ARI Events: {APP_NAME}")

def run():
    cleanup_all()
    ws_url = f"{WS_URL}?api_key={USER}:{PASSWORD}&app={APP_NAME}"
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever(ping_interval=30, ping_timeout=10)

