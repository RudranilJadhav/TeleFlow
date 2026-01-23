import websocket
import base64

USER="6001"
PASS="unsecurepassword"
STREAM_APP="stream-app"

ws = websocket.WebSocket()
ws.connect(
    "ws://localhost:8088/ari/events"
    "?app={STREAM_APP}"
    "&api_key={USER}:{PASS}"
)
