import re
from multiprocessing import Queue
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Model selection - you can change this as needed
MODEL = "llama-3.3-70b-versatile"  # or "mixtral-8x7b-32768", "gemma2-9b-it", etc.

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

# Load system prompt
with open("../utils/prompt.txt", "r") as f:
    system_prompt = f.read()

print(f"Loaded system prompt, using Groq model: {MODEL}")

# Initialize message history
messages = [
    {"role": "system", "content": system_prompt}
]

def run_llm(text_queue, out_queue, user_speaking_event, ai_speaking_event, transcript_queue):
    global messages
    
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
                max_tokens=100,  # Slightly higher than before for complete sentences
                top_p=0.85,
                stream=True,
                stop=["\nUser:", "*"]  # Stop sequences
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
                    output = clean_llm_output(chunk.choices[0].delta.content)
                    
                    if not output.strip():
                        continue
                    
                    assistant_response += output
                    sentence_buffer += output
                    
                    # Send complete sentences to TTS
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
        
        # Keep message history manageable (prevent token limit issues)
        if len(messages) > 20:  # Adjust based on your needs
            # Keep system message and last 19 exchanges
            messages = [messages[0]] + messages[-19:]