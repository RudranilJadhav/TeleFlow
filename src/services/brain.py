import re
from multiprocessing import Queue
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "llama-3.3-70b-versatile"

def run_llm(text_queue, out_queue, user_speaking_event, ai_speaking_event, transcript_queue,type):
    
    global messages

    if type=="Outbound":
        with open("../utils/outboundprompt.txt", "r") as f:
            system_prompt = f.read()
    if type=="Inbound":
        with open("../utils/inboundprompt.txt", "r") as f:
            system_prompt = f.read()

    messages = [
    {"role": "system", "content": system_prompt}
    ]
    print(f"Loaded system prompt, using Groq model: {MODEL}")
    while True:
        text = text_queue.get()
        if text is None:
            break
            
        print("\nUser:", text)
        transcript_queue.put(f"User: {text}")
        # Add user message to history
        messages.append({"role": "user", "content": text})
        
        ai_speaking_event.set()
        assistant_response = ""
        sentence_buffer = ""
        
        try:
            # Create streaming completion
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.5,
                max_tokens=100,
                top_p=0.85,
                stream=True,
            )
            
            # Process stream chunks
            for chunk in stream:
                # Check for barge-in
                if user_speaking_event.is_set():
                    print("LLM interrupted")
                    ai_speaking_event.clear()
                    # Clear TTS queue
                    while not out_queue.empty():
                        try:
                            out_queue.get_nowait()
                        except:
                            break
                    break
                
                # Get content from chunk
                if chunk.choices[0].delta.content:
                    # output = clean_llm_output(chunk.choices[0].delta.content)
                    output = chunk.choices[0].delta.content
                    
                    if not output.strip():
                        continue
                    
                    assistant_response += output
                    sentence_buffer += output
                    if re.search(r'[.!?](?:\s|$)', sentence_buffer):
                        out_queue.put(sentence_buffer.strip())
                        sentence_buffer = ""
            
            # Send any remaining buffer
            if sentence_buffer and not user_speaking_event.is_set():
                out_queue.put(sentence_buffer.strip())
            # Add assistant response to history if we weren't interrupted
            if assistant_response and not user_speaking_event.is_set():
                messages.append({"role": "assistant", "content": assistant_response})
                transcript_queue.put(f"Agent: {assistant_response}")
            
        except Exception as e:
            print(f"Error in LLM stream: {e}")
            # Fallback response
            fallback = "I'm sorry, I'm having trouble processing that right now."
            out_queue.put(fallback)
            messages.append({"role": "assistant", "content": fallback})
            transcript_queue.put(f"Agent: {fallback}")
        
        ai_speaking_event.clear()
        
        # Keep message history manageable to prevent token limit issues
        if len(messages) > 20:
            messages = [messages[0]] + messages[-19:]