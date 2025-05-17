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

router = APIRouter(prefix="/api/llm_generate", tags=["llm_generate"])

class TTSRequest(BaseModel):
    text: str

@router.post("/text")
async def process_text(request: TTSRequest):
    """Process text input and return LLM text response"""
    try:
        # Initialize client
        client = OpenAI(
            api_key=os.getenv("DASH_SCOPE_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        
        # Call API with text-only response
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "user", "content": request.text}
            ],
            # No audio modalities here
            temperature=0.7,
            max_tokens=500
        )

        # Extract response text
        response_text = completion.choices[0].message.content
        
        # Return as JSON
        return JSONResponse({
            "response": {
                "text": response_text,
                "model": "qwen-plus"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))