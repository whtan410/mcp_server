import os
import base64
import numpy as np
import soundfile as sf
from openai import OpenAI

def text_to_speech(text, output_file=None, voice="Chelsie"):
    """
    Convert text to speech using DashScope API
    
    Args:
        text (str): Text to convert to speech
        output_file (str, optional): Path to save audio file (if None, won't save)
        voice (str, optional): Voice to use (default: "Chelsie")
        
    Returns:
        numpy.ndarray: Audio data as numpy array
    """
    # Initialize client
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    
    # Call API with audio generation - using system message to constrain behavior
    completion = client.chat.completions.create(
        model="qwen2.5-omni-7b",
        messages=[
            {"role": "system", "content": "You are a text-to-speech engine. Convert the user's text to speech without adding, removing, or changing anything. Do not provide any commentary, explanations, or additional text. Only process the exact text as provided."},
            {"role": "user", "content": text}
        ],
        modalities=["text", "audio"],
        audio={"voice": voice, "format": "wav"},
        stream=True,
        stream_options={"include_usage": True},
    )
    
    # Collect audio data
    audio_string = ""
    
    for chunk in completion:
        if chunk.choices and hasattr(chunk.choices[0].delta, "audio"):
            try:
                audio_string += chunk.choices[0].delta.audio["data"]
            except Exception:
                pass
    
    # Convert base64 string to audio numpy array
    wav_bytes = base64.b64decode(audio_string)
    audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
    
    # Save to file if output_file is provided
    if output_file:
        sf.write(output_file, audio_np, samplerate=24000)
        
    return audio_np

# Example usage:
if __name__ == "__main__":
    audio_np = text_to_speech(
        "What is solar system?",
        output_file="speech_output.wav"
    )