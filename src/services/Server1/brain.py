def run_llm(text_queue):
    while True:
        text = text_queue.get()   # BLOCKS
        print("LLM got:", text)

        # ---- LLM LOGIC HERE ----
