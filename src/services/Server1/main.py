import multiprocessing
from multiprocessing import Process, Queue

from brain import run_llm
from events import run as run_events
from external_media import stream_to_whisper

if __name__ == "__main__":
    
    text_queue = Queue()

    p_asr = Process(
        target=stream_to_whisper,
        args=(text_queue,),
        name="ASR"
    )

    p_llm = Process(
        target=run_llm,
        args=(text_queue,),
        name="LLM"
    )

    p_events = Process(
        target=run_events,
        name="EVENTS"
    )
    p_events.start()
    p_asr.start()
    p_llm.start()
    p_asr.join()
    p_llm.join()
    p_events.join()
