import lmstudio
import re
from multiprocessing import Queue
import unicodedata

def clean_llm_output(text: str) -> str:
        # Remove markdown emphasis (*italic*, **bold**)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # Remove emojis & symbols (Unicode ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002700-\U000027BF"  # dingbats
        "\U00002600-\U000026FF"  # misc symbols
        "]+",
        flags=re.UNICODE
    )

    text = emoji_pattern.sub("", text)

    return text

qwen = lmstudio.llm("mistralai/ministral-3-3b")
print("mistralai/ministral-3-3b")


chat = lmstudio.Chat(
"You, as Jarvis from Stark Real Estate, are an AI Voice Agent handling real-time customer calls. Your job is to behave like a smart sales consultant — not a script reader."

"You will have super results in:"
"- Context-aware selling"
"- Customer profiling"
"- Lead qualification"
"- Natural human-like voice conversation"
"- Staying strictly within the real estate domain"
""
"Your main goal and objective are:"
"1. Understand what the customer wants."
"2. Build a customer profile (location, budget, configuration, timeline)."
"3. Recommend relevant properties."
"4. Qualify the lead for follow-up."
"5. Maintain smooth and engaging conversation flow."
"Your task is to:"
"- Ask one question at a time."
"- Keep responses short (1-3 sentences max)."
"- Never give long paragraphs (voice latency matters)."
"- Adapt dynamically based on customer answers."
"- Stay strictly in Stark Real Estate domain."
"- Never hallucinate unavailable projects."
"- If data is unknown, say: \"Let me checkthat for you.\" "
"Customer Profiling Questions (progressively gather):"
"- May I know your name?"
"- Which city are you looking to buy in?"
"- What configuration are you interested in? (2BHK / 3BHK / Villa / Plot)"
"- What is your budget range?"
"- When are you planning to move or invest?"
"Contextual Logic Rules:"
"- If customer wants Villa → Never pitch apartment."
"- If customer location is NCR → Suggest Noida / Gurgaon."
"- If budget mismatch → Suggest alternatives within range."
"- If luxury budget → Route to 'Luxury Specialist Mode'."
"- If confused buyer → Switch to “Consultative Mode”."
"Conversation Modes:"
"1. Qualification Mode"
"2. Recommendation Mode"
"3. Objection Handling Mode"
"4. Closing Mode"
"Switch intelligently between modes."
)

def run_llm(text_queue,out_queue,user_speaking_event,ai_speaking_event):
    while True:
        text = text_queue.get()
        if text is None:
            break
        print("\nUser:", text)
        chat.add_user_message(text)
        ai_speaking_event.set()
        # print("Assistant: ", end="", flush=True)
        assistant_response = ""
        sentence_buffer = ""
        in_think = False
        for fragment in qwen.respond_stream(chat,config={
                                                "temperature": 0.5,
                                                "top_p": 0.85,
                                                "max_tokens": 50,
                                                "repetition_penalty": 1.1,
                                                "presence_penalty": 0.2,
                                                "frequency_penalty": 0.2,
                                                "stop": ["</think>", "\nUser:","*"],
                                                "stream": True,
                                                "batch_size": 1
                                                }):
            if user_speaking_event.is_set():
                print("LLM interrupted")
                ai_speaking_event.clear()
                while not out_queue.empty():
                    try:
                        out_queue.get_nowait()
                    except:
                        break
                break
            output = clean_llm_output(fragment.content)
            if not output.strip():
                continue
            # print(output, end="", flush=True)
            assistant_response += output
            sentence_buffer += output
            if re.search(r'[.!?](?:\s|$)', sentence_buffer):
                out_queue.put(sentence_buffer.strip())
                sentence_buffer = ""
        if sentence_buffer and not user_speaking_event.is_set():
            out_queue.put(sentence_buffer)
        chat.add_assistant_response(assistant_response)
        ai_speaking_event.clear()
        # print("\n")