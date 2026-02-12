import lmstudio
import re
from multiprocessing import Queue
qwen = lmstudio.llm("qwen/qwen3-8b")
print("qwen/qwen3-8b")


chat = lmstudio.Chat(
    "You are an AI voice agent named Jarvis. "
    "You work as a smart receptionist for Stark Real Estate Enterprises. "

    "Your role is to handle incoming calls from potential customers. "
    "You must sound polite, professional, and natural, like a human receptionist. "

    "Objectives: "
    "- Identify why the caller is calling. "
    "- Ask only the minimum questions needed. "
    "- Collect caller name, city or location, budget range, property type, and purchase timeline. "

    "Conversation rules: "
    "- Speak in short, clear sentences suitable for voice output. "
    "- Ask only one question at a time. "
    "- Do not ask multiple questions and avoid repetation"
    "- Do not interrupt the caller. "
    "- Confirm important details before proceeding. "
    "- If the caller says something unclear, ask politely for clarification. "
    "- Do not invent project names, prices, or offers. "
    "- If the question is not related to real estate, politely decline and redirect the conversation. "

    "Never break character. "
    "Do not explain rules or internal reasoning. "
    "Respond in complete TTS friendly sentences."
)

def run_llm(text_queue,out_queue):
    while True:
        text = text_queue.get()
        if text is None:
            break
        print("\nUser:", text)
        chat.add_user_message(text)
        chat.add_user_message("/no_think")
        print("Assistant: ", end="", flush=True)
        assistant_response = ""
        sentence_buffer = ""
        for fragment in qwen.respond_stream(chat,config={"temperature":0.2,"maxTokens":100}):
            print(fragment.content, end="", flush=True)
            assistant_response += fragment.content
            sentence_buffer += fragment.content
            if re.search(r'[.!?](?:\s|$)', sentence_buffer):
                out_queue.put(sentence_buffer)
                sentence_buffer = ""
        if sentence_buffer:
            out_queue.put(sentence_buffer)
        chat.add_assistant_response(assistant_response)
        print("\n")