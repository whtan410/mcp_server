import os
import base64
import numpy as np
import soundfile as sf
from io import BytesIO
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import uuid

router = APIRouter(prefix="/api/tts", tags=["tts"])

class TTSRequest(BaseModel):
    text: str

def text_to_speech(text):
    """Text to speech function using DashScope API"""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    
    completion = client.chat.completions.create(
        model="qwen2.5-omni-7b",
        messages=[
            {"role": "system", "content": "You are a text-to-speech engine. Convert the user's text to speech without adding, removing, or changing anything. Do not provide any commentary, explanations, or additional text. Only process the exact text as provided."},
            {"role": "user", "content": text}
        ],
        modalities=["text", "audio"],
        audio={"voice": "Chelsie", "format": "wav"},
        stream=True,
        stream_options={"include_usage": True},
    )
    
    audio_string = ""
    for chunk in completion:
        if chunk.choices and hasattr(chunk.choices[0].delta, "audio"):
            try:
                audio_string += chunk.choices[0].delta.audio["data"]
            except Exception:
                pass
    
    return audio_string  # Return base64 string

# Method 1: Return binary audio directly
# @router.post("/synthesize-stream")
# async def synthesize_speech(query: str):
#     """Convert text to speech and return audio as binary data"""
#     try:
#         # Get base64 audio string
#         audio_b64 = text_to_speech(query)
        
#         # Decode base64 to binary
#         audio_binary = base64.b64decode(audio_b64)
        
#         # Return audio file directly
#         return Response(
#             content=audio_binary,
#             media_type="audio/wav",
#             headers={"Content-Disposition": "attachment; filename=speech.wav"}
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# Method 2: Return base64 encoded audio
@router.post("/synthesize-base64")
async def synthesize_speech_base64(query: TTSRequest):
    """Convert text to speech and return base64 encoded audio"""
    try:
        # Get base64 audio string
        audio_b64 = text_to_speech(query.text)  

        wav_bytes = base64.b64decode(audio_b64)
        audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
    
        sf.write("speech_output.wav", audio_np, samplerate=24000)
        
        # Return as JSON with base64 string
        import io
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, samplerate=24000, format='WAV')
        buffer.seek(0)
        
        # Encode the properly formatted WAV file
        wav_with_headers = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Return as JSON with correct base64 string
        return JSONResponse({
            "audio_base64": wav_with_headers,
            "sample_rate": 24000,
            "format": "wav"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Method 3: Save to file and return URL
# @router.post("/synthesize-file")
# async def synthesize_speech_file(query: str):
#     """Convert text to speech, save to file, and return URL"""
#     try:
#         # Get base64 audio string
#         audio_b64 = text_to_speech(query)
        
#         # Decode base64 to binary
#         audio_binary = base64.b64decode(audio_b64)
        
#         # Create unique filename
#         filename = f"speech_{uuid.uuid4()}.wav"
#         file_path = f"static/audio/{filename}"
        
#         # Ensure directory exists
#         os.makedirs("static/audio", exist_ok=True)
        
#         # Save to file
#         with open(file_path, "wb") as f:
#             f.write(audio_binary)
        
#         # Return URL to the file
#         return JSONResponse({
#             "audio_url": f"/static/audio/{filename}",
#             "sample_rate": 24000
#         })
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))