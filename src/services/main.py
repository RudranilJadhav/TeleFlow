from multiprocessing import Process, Queue , Event
import threading
from brain import run_llm
from events import run as run_events
from external_media import stream_to_whisper
from tts import run_piper

if __name__ == "__main__":

    text_queue = Queue()
    out_queue = Queue()
    transcript_queue = Queue()
    text_queue.put("Hello")

    #shared events
    user_speaking_event = Event()
    ai_speaking_event = Event()

    threading.Thread(
        target=run_llm,
        args=(text_queue,out_queue,user_speaking_event,ai_speaking_event,transcript_queue),
        daemon=True
    ).start()

    p_asr = Process(
        target=stream_to_whisper,
        args=(text_queue,out_queue,user_speaking_event,ai_speaking_event),
        name="ASR"
    )

    p_events = Process(
        target=run_events,
        args=(transcript_queue,),
        name="EVENTS"
    )

    p_piper = Process(
        target=run_piper,
        args=(out_queue,user_speaking_event,ai_speaking_event),
        name="PIPER"
    )
    
    p_events.start()
    p_asr.start()
    p_piper.start()

    p_piper.join()
    p_asr.join()
    p_events.join()