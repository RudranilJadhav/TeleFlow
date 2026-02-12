from multiprocessing import Process, Queue
import threading

from brain import run_llm
from events import run as run_events
from external_media import stream_to_whisper
from tts import run_piper

if __name__ == "__main__":

    text_queue = Queue()
    out_queue = Queue()

    threading.Thread(
        target=run_llm,
        args=(text_queue,out_queue),
        daemon=True
    ).start()

    p_asr = Process(
        target=stream_to_whisper,
        args=(text_queue,),
        name="ASR"
    )

    p_events = Process(
        target=run_events,
        name="EVENTS"
    )

    p_piper = Process(
        target=run_piper,
        args=(out_queue,),
        name="PIPER"
    )
    
    p_events.start()
    p_asr.start()
    p_piper.start()

    p_piper.join()
    p_asr.join()
    p_events.join()
