import asyncio
import io
import os
import random
import tempfile
import uuid
from datetime import datetime
from typing import List, Dict, Any
import base64

import numpy as np
import cv2
from PIL import Image
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel



app = FastAPI(title="Langate Story Generator API")

# Global variables for API clients
openai_client = None
hf_client = None
# hf_headers = None
elevenlabs_headers = None

class StoryRequest(BaseModel):
    weather: str = "foggy"
    length: int = 200
    voice: str = "af_heart"

class StoryResponse(BaseModel):
    audio_files: List[str]  # Base64 encoded audio files
    text: str
    event: str
    processing_time: str

# API clients - no need for global model loading
openai_client = None
hf_client =  None
# hf_headers = None
elevenlabs_headers = None

# Startup event to initialize API clients
@app.on_event("startup")
async def initialize_apis():
    global openai_client,hf_client , elevenlabs_headers, #hf_headers
    try:
        # Initialize OpenAI client
        import openai
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize Hugging Face headers
        from huggingface_hub import InferenceClient
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not hf_api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set")
        # hf_headers = {"Authorization": f"Bearer {hf_api_key}"} #not using the huggingface_hub lib
        hf_client = InferenceClient(
            provider="auto",
            api_key=os.environ[hf_api_key],
        )
  

        # Initialize ElevenLabs headers
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")
        elevenlabs_headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_api_key
        }
        
        print("API clients initialized successfully")
        print(f"OpenAI client ready: {openai_client is not None}")
        # print(f"Hugging Face headers ready: {hf_headers is not None}")
        print(f"Hugging Face headers ready: {hf_client is not None}")
        print(f"ElevenLabs headers ready: {elevenlabs_headers is not None}")
        
    except Exception as e:
        print(f"Error initializing API clients: {e}")
        raise e

def parse_model_output(output: str) -> Dict[str, str]:
    """Parse model output into dictionary."""
    return {
        str(i): line.split(". ", 1)[-1]
        for i, line in enumerate(output.strip().split("\n"), start=1)
    }

def generate_event(photo_elements: Dict[str, str]) -> str:
    """Generate random event based on photo elements."""
    subjects = ["Strange amphibian", "major", "not so secret disposal company", "crazy duck", "very normal alien"]
    verbs = ["jumps over", "solves", "paints", "explores", "repairs", "builds", "eats", "boils"]
    adjectives = ["lazy", "mysterious", "vibrant", "ancient", "futuristic", "dark"]
    
    element = random.choice(list(photo_elements.values())) if photo_elements else "object"
    return f"A {random.choice(subjects)} {random.choice(verbs)} a {random.choice(adjectives)} {element}."

def generate_prompt(event: str, weather: str, calendar: datetime, length: int) -> str:
    """Generate prompt for story generation."""
    dynamic_prompt = f"Setting: Langate, {calendar.month}/{calendar.day}, {weather}. Event: {event}. Create a {length}-word real-time report on this event. "
    base = f"""
        Create a story in present tense like it's being told by a radio community announcement host who's in the town of Langate. Act calm, and largely unbothered by supernatural happenings.
        Report in present tense on today's {calendar.month}/{calendar.day} terrifying or absurd events in a dry, eerie tone laced with dark humor.
    """
    return f"{base}{dynamic_prompt}"

def clean_text(text: str) -> str:
    """Clean text for processing."""
    return ' '.join(line.strip() for line in text.splitlines() if line.strip())

async def analyze_image_with_api(image_data: bytes) -> Dict[str, str]:
    """
    Analyze image using OpenAI Vision API.
    """
    global openai_client
    
    if not openai_client:
        print("OpenAI client not available, using fallback")
        return {"1": "building", "2": "tree", "3": "sky"}
    
    try:
        # Convert image to base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "List three different elements of this image in order of distance from the viewer. Format as: 1. [element], 2. [element], 3. [element]"
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }
                ]
            }],
            max_tokens=150
        )
        
        return parse_model_output(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        # Return fallback elements
        return {"1": "building", "2": "tree", "3": "sky"}

async def generate_story_with_api(prompt: str) -> str:
    """
    Generate story using Hugging Face Inference API.
    """
    global hf_client #hf_headers
    
    # if not hf_headers:
    #     print("Hugging Face client not available, using fallback")
    #     return "In the misty town of Langate today, residents report unusual occurrences involving local wildlife and mysterious structures. The mayor assures everyone this is perfectly normal for a Tuesday."
    
    # try:
    #     import aiohttp
        
    #     API_URL = "https://router.huggingface.co/hf-inference/models/sarvamai/sarvam-m"

    #     # def query(payload):
    #     #     response = requests.post(API_URL, headers=headers, json=payload)
    #     #     return response.json()

    #     # output = query({
    #     #     "inputs": "Can you please let us know more details about your ",
    #     # })

    #     payload = {
    #         "inputs": prompt,
    #         "parameters": {
    #             "max_length": 500,
    #             "temperature": 0.7,
    #             "do_sample": True
    #         }
    #     }
        
    #     async with aiohttp.ClientSession() as session:
    #         async with session.post(API_URL, headers=hf_headers, json=payload) as response:
    #             if response.status == 200:
    #                 result = await response.json()
    #                 if isinstance(result, list) and len(result) > 0:
    #                     # return result[0].get('generated_text', prompt)
    #                     # return result.json()
    #                     return result["choices"][0]["message"]
    #                 else:
    #                     return "In the misty town of Langate today, residents report unusual occurrences. The mayor assures everyone this is perfectly normal."
    #             else:
    #                 print(f"HF API error: {response.status}")
    #                 return "In the misty town of Langate today, residents report unusual occurrences. The mayor assures everyone this is perfectly normal."
                    
    # except Exception as e:
    #     print(f"Error generating story: {e}")
    #     return "In the misty town of Langate today, residents report unusual occurrences involving local wildlife and mysterious structures. The mayor assures everyone this is perfectly normal for a Tuesday."

    if not hf_client:
        print("Hugging Face client not available, using fallback")
        return "In the misty town of Langate today, residents report unusual occurrences involving local wildlife and mysterious structures. The mayor assures everyone this is perfectly normal for a Tuesday."
        
    try:
        completion = hf_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        return completion.choices[0].message
    
    except Exception as e:
        print(f"Error generating story: {e}")
        return "In the misty town of Langate today, residents report unusual occurrences involving local wildlife and mysterious structures. The mayor assures everyone this is perfectly normal for a Tuesday."

async def generate_audio_with_api(text: str, voice: str) -> List[bytes]:
    """
    Generate audio using ElevenLabs API.
    """
    global elevenlabs_headers
    
    if not elevenlabs_headers:
        print("ElevenLabs client not available, using fallback")
        return [create_fallback_audio()]
    
    # Voice mapping (you can customize these with your ElevenLabs voice IDs)
    voice_map = {
        "af_heart": "21m00Tcm4TlvDq8ikWAM",  # Rachel
        "default": "21m00Tcm4TlvDq8ikWAM",
        "male": "29vD33N1CtxCmqQRPOHJ",      # Drew
        "female": "21m00Tcm4TlvDq8ikWAM"     # Rachel
    }
    
    voice_id = voice_map.get(voice, voice_map["default"])
    
    try:
        import aiohttp
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=elevenlabs_headers, json=payload) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    return [audio_data]
                else:
                    print(f"ElevenLabs API error: {response.status}")
                    error_text = await response.text()
                    print(f"Error details: {error_text}")
                    # Return fallback audio (silence)
                    return [create_fallback_audio()]
                    
    except Exception as e:
        print(f"Error generating audio: {e}")
        return [create_fallback_audio()]

def create_fallback_audio() -> bytes:
    """Create a fallback audio file (1 second of silence)."""
    try:
        silence = np.zeros(24000, dtype=np.float32)  # 1 second at 24kHz
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, silence, 24000, format='WAV')
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception:
        return b''  # Empty bytes as last resort

@app.post("/generate-story", response_model=StoryResponse)
async def generate_story_from_image(
    file: UploadFile = File(...),
    weather: str = "foggy",
    length: int = 200,
    voice: str = "af_heart"
):
    """
    Generate a Langate story from an uploaded image.
    """
    start_time = datetime.now()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Validate image size (limit to 10MB)
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        # Process image
        try:
            image = Image.open(io.BytesIO(image_data))
            # Resize for processing efficiency
            image = image.resize((image.width // 4, image.height // 4), Image.LANCZOS)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Analyze image (replace with cloud API call)
        photo_elements = await analyze_image_with_api(image_data)
        
        # Generate event and story
        event = generate_event(photo_elements)
        prompt = generate_prompt(event, weather, datetime.now(), length)
        
        # Generate story text (replace with cloud API call)
        story_text = await generate_story_with_api(prompt)
        clean_story = clean_text(story_text)
        
        # Generate audio (replace with cloud API call)
        audio_bytes_list = await generate_audio_with_api(clean_story, voice)
        
        # Convert audio to base64 for JSON response
        audio_files_b64 = []
        for i, audio_bytes in enumerate(audio_bytes_list):
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            audio_files_b64.append(audio_b64)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = str(end_time - start_time)
        
        return StoryResponse(
            audio_files=audio_files_b64,
            text=clean_story,
            event=event,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Langate Story Generator API",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate-story": "Generate story from image",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))