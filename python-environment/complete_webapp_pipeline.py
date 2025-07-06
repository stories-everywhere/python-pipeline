import asyncio
import io
import os
import random
import tempfile
import uuid
from datetime import datetime
from typing import List, Dict, Any
import base64
from contextlib import asynccontextmanager

import numpy as np
import cv2
from PIL import Image
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import aiohttp

# from huggingface_hub import InferenceClient
import fal_client
import moondream as md
from openai import OpenAI


# Global variables for API clients
openai_client = None
# hf_client = None
elevenlabs_headers = None
fal_client_instance = None
md_client = None


class StoryRequest(BaseModel):
    """Request model for story generation parameters."""
    weather: str = "foggy"
    length: int = 200
    voice: str = "af_heart"

class StoryResponse(BaseModel):
    """Response model for generated story data."""
    audio_files: List[str]  # Base64 encoded audio files
    text: str
    event: str
    processing_time: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI application.
    Initializes API clients on startup and handles cleanup on shutdown.
    """
    # Startup - Initialize API clients
    global openai_client, elevenlabs_headers, fal_client_instance, md_client # hf_client,
    try:
        # Initialize OpenAI client (currently commented out)
        # import openai
        sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")
        # if not openai_api_key:
        #     raise ValueError("OPENAI_API_KEY environment variable not set")
        # openai_client = openai.OpenAI(api_key=openai_api_key)

        openai_client = OpenAI(
            api_key=sambanova_api_key,
            base_url="https://api.sambanova.ai/v1",
        )
        
        # Initialize Hugging Face client for text generation and image analysis
        # hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        # if not hf_api_key:
        #     raise ValueError("HUGGINGFACE_API_KEY environment variable not set")
        
        # hf_client = InferenceClient(
        #     provider="auto",
        #     api_key=hf_api_key,
        # )

        # Verify FAL API key is set (fal_client handles the key automatically)
        fal_api_key = os.getenv("FAL_KEY")
        if not fal_api_key:
            raise ValueError("FAL_KEY environment variable not set")
        

        md_api_key = os.getenv("MOONDREAM_API_KEY")
        if not md_api_key:
            raise ValueError("MOONDREAM_API_KEY environment variable not set")
        
        md_client = md.vl(
            api_key=md_api_key,
        )

        
        # Initialize ElevenLabs headers (currently commented out)
        # elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        # if not elevenlabs_api_key:
        #     raise ValueError("ELEVENLABS_API_KEY environment variable not set")
        # elevenlabs_headers = {
        #     "Accept": "audio/mpeg",
        #     "Content-Type": "application/json",
        #     "xi-api-key": elevenlabs_api_key
        # }
        
        print("API clients initialized successfully")
        # print(f"Hugging Face client ready: {hf_client is not None}")
        
    except Exception as e:
        print(f"Error initializing API clients: {e}")
        raise e
    
    yield
    
    # Shutdown - cleanup if needed
    pass

# Initialize FastAPI app with lifespan handler
app = FastAPI(title="Langate Story Generator API", lifespan=lifespan)

def parse_model_output(output: str) -> Dict[str, str]:
    """
    Parse model output into a dictionary format.
    
    Args:
        output: Raw model output string
        
    Returns:
        Dictionary mapping indices to parsed elements
    """
    return {
        str(i): line.split(". ", 1)[-1]
        for i, line in enumerate(output.strip().split("\n"), start=1)
    }

def generate_event(photo_elements: Dict[str, str]) -> str:
    """
    Generate a random surreal event based on photo elements.
    
    Args:
        photo_elements: Dictionary of elements identified in the photo
        
    Returns:
        A randomly generated event description
    """
    subjects = [
        "Strange amphibian", "major", "not so secret disposal company", 
        "crazy duck", "very normal alien"
    ]
    verbs = [
        "jumps over", "solves", "paints", "explores", 
        "repairs", "builds", "eats", "boils"
    ]
    adjectives = [
        "lazy", "mysterious", "vibrant", "ancient", 
        "futuristic", "dark"
    ]
    
    # Select a random element from the photo or use default
    element = random.choice(list(photo_elements.values())) if photo_elements else "object"
    
    return f"A {random.choice(subjects)} {random.choice(verbs)} a {random.choice(adjectives)} {element}."

def generate_prompt(event: str, weather: str, calendar: datetime, length: int) -> str:
    """
    Generate a prompt for story generation based on parameters.
    
    Args:
        event: The generated event description
        weather: Current weather condition
        calendar: Current datetime
        length: Desired story length in words
        
    Returns:
        Formatted prompt string for the AI model
    """
    dynamic_prompt = (
        f"Setting: Langate, {calendar.month}/{calendar.day}, {weather}. "
        f"Event: {event}. Create a {length}-word real-time report on this event. "
    )
    
    base_prompt = """
        Create a story in present tense like it's being told by a radio community 
        announcement host who's in the town of Langate. Act calm, and largely 
        unbothered by supernatural happenings. Report in present tense on today's 
        {}/{} terrifying or absurd events in a dry, eerie tone laced with dark humor.
    """.format(calendar.month, calendar.day)
    
    return f"{base_prompt.strip()}{dynamic_prompt}"

def clean_text(text: str) -> str:
    """
    Clean and format text for processing.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text with proper spacing
    """
    return ' '.join(line.strip() for line in text.splitlines() if line.strip())

async def analyze_image_with_api(image_data: bytes) -> Dict[str, str]:
    """
    Analyze an image using Hugging Face Vision API to identify elements.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Dictionary of identified image elements
    """
    # global hf_client
    # if not hf_client:
    #     print("HF client not available, using fallback")
    #     return {"1": "building", "2": "tree", "3": "sky"}

    # try:
    #     # Convert image to base64 for API transmission
    #     image_b64 = base64.b64encode(image_data).decode('utf-8')
        
    #     # Make API call to vision model
    #     completion = hf_client.chat.completions.create(
    #         model="Qwen/Qwen2.5-VL-32B-Instruct",
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "text",
    #                         "text": (
    #                             "List three different elements of this image in order "
    #                             "of distance from the viewer. Format as: "
    #                             "1. [element], 2. [element], 3. [element]"
    #                         )
    #                     },
    #                     {
    #                         "type": "image_url",
    #                         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
    #                     }
    #                 ]
    #             }
    #         ],
    #     )

    #     # Extract and parse the response
    #     response_content = completion.choices[0].message.content
    #     return parse_model_output(response_content)
        

    global md_client
    if not md_client:
        print("MoonDream client not available, using fallback")
        return {"1": "building", "2": "tree", "3": "sky"}

    try:
        # Convert image to base64 for API transmission
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        # Make API call to vision model
        response_content = md_client.query(f"data:image/jpeg;base64,{image_b64}", (
                                "List three different elements of this image in order of distance from the viewer. Format as: 1. [element], 2. [element], 3. [element]"
                            ))['answer']

        # Parse the response
        return parse_model_output(response_content)

    except Exception as e:
        print(f"Error analyzing image: {e}")
        # Return fallback elements on error
        return {"1": "building", "2": "tree", "3": "sky"}

async def generate_story_with_api(prompt: str) -> str:
    """
    Generate a story using Hugging Face text generation API.
    
    Args:
        prompt: The story generation prompt
        
    Returns:
        Generated story text
    """
    # global hf_client
    
    # if not hf_client:
    #     print("Hugging Face client not available, using fallback")
    #     return (
    #         "In the misty town of Langate today, residents report unusual "
    #         "occurrences involving local wildlife and mysterious structures. "
    #         "The mayor assures everyone this is perfectly normal for a Tuesday."
    #     )
        

        
    # try:
    #     # Generate story using chat completion API
    #     completion = hf_client.chat.completions.create(
    #         model="deepseek-ai/DeepSeek-V3-0324",
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": prompt
    #             }
    #         ],
    #     )
        

    #     # Extract the generated content
    #     return completion.choices[0].message.content

    global openai_client
    if not openai_client:
        print("Open Ai [sambanova] client not available, using fallback")
        return (
            "In the misty town of Langate today, residents report unusual "
            "occurrences involving local wildlife and mysterious structures. "
            "The mayor assures everyone this is perfectly normal for a Tuesday."
        )
    
    try:
        response = openai_client.chat.completions.create(
            model="DeepSeek-V3-0324",
            messages=[
                # {
                #     "role":"system",
                #     "content":"You are a helpful assistant"
                # },
                {
                    "role":"user",
                    "content":prompt
                }
            ],
            temperature=0.1,
            top_p=0.1
        )

        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error generating story: {e}")
        return (
            "In the misty town of Langate today, residents report unusual "
            "occurrences involving local wildlife and mysterious structures. "
            "The mayor assures everyone this is perfectly normal for a Tuesday."
        )

async def generate_audio_with_api(text: str, voice: str) -> List[bytes]:
    """
    Generate audio from text using FAL AI text-to-speech API.
    
    Args:
        text: Text to convert to speech
        voice: Voice identifier (currently unused)
        
    Returns:
        List of audio bytes
    """
    try:
        # Submit async request to FAL AI TTS service
        handler = await fal_client.submit_async(
            "fal-ai/chatterbox/text-to-speech",
            arguments={
                "audio_url": "https://v3.fal.media/files/elephant/wLS77pG8fFjdqybPlKp3g_1-common_voice_en_39613299.mp3",
                "exaggeration": 0.3,
                "temperature": 0.7,
                "cfg": 0.5,
                "text": text
            },
        )

        # Monitor the processing with logs (commented for too many logs)
        # async for event in handler.iter_events(with_logs=True):
            # print(f"TTS Event: {event}")
            
        # Get the final result
        result = await handler.get()
        
        # Extract audio URL from the FAL API response
        # Response format: {"audio": {"url": "...", "content_type": "...", ...}}
        if "audio" in result and "url" in result["audio"]:
            audio_url = result["audio"]["url"]
            print(f"Generated audio URL: {audio_url}")
            
            # Fetch the audio file from the URL
            audio_bytes = await fetch_audio_from_url(audio_url)
            
            if audio_bytes:
                return [audio_bytes]
            else:
                print("Failed to fetch audio from URL, using fallback")
                return [create_fallback_audio()]
        else:
            print("No audio URL found in FAL API response, using fallback")
            return [create_fallback_audio()]
             
    except Exception as e:
        print(f"Error generating audio: {e}")
        return [create_fallback_audio()]

def create_fallback_audio() -> bytes:
    """
    Create a fallback audio file (1 second of silence) when TTS fails.
    
    Returns:
        Audio bytes representing 1 second of silence
    """
    try:
        # Generate 1 second of silence at 24kHz
        silence = np.zeros(24000, dtype=np.float32)
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, silence, 24000, format='WAV')
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        print(f"Error creating fallback audio: {e}")
        return b''  # Return empty bytes as last resort

async def fetch_audio_from_url(url: str) -> bytes:
    """
    Fetch audio file from a URL and return as bytes.
    
    Args:
        url: URL of the audio file to fetch
        
    Returns:
        Audio file as bytes, or empty bytes if fetch fails
    """
    try:
        timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    audio_bytes = await response.read()
                    print(f"Successfully fetched {len(audio_bytes)} bytes from {url}")
                    return audio_bytes
                else:
                    print(f"Failed to fetch audio: HTTP {response.status}")
                    return b''
    except asyncio.TimeoutError:
        print(f"Timeout fetching audio from {url}")
        return b''
    except Exception as e:
        print(f"Error fetching audio from {url}: {e}")
        return b''

@app.post("/generate-story", response_model=StoryResponse)
async def generate_story_from_image(
    file: UploadFile = File(...),
    weather: str = "foggy",
    length: int = 200,
    voice: str = "af_heart"
):
    """
    Generate a Langate story from an uploaded image.
    
    Args:
        file: Uploaded image file
        weather: Weather condition for the story setting
        length: Desired story length in words
        voice: Voice identifier for TTS (currently unused)
        
    Returns:
        StoryResponse containing generated story, audio, and metadata
    """
    start_time = datetime.now()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate image data
        image_data = await file.read()
        
        # Validate image size (limit to 10MB)
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        # Process and resize image for efficiency
        try:
            image = Image.open(io.BytesIO(image_data))
            # Resize to 1/4 size for faster processing
            image = image.resize(
                (image.width // 4, image.height // 4), 
                Image.LANCZOS
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid image format: {str(e)}"
            )
        
        # Analyze image to extract elements
        photo_elements = await analyze_image_with_api(image_data)
        
        # Generate event and story
        event = generate_event(photo_elements)
        prompt = generate_prompt(event, weather, datetime.now(), length)
        
        # Generate story text
        story_text = await generate_story_with_api(prompt)
        clean_story = clean_text(story_text)
        
        # Generate audio from story text
        audio_bytes_list = await generate_audio_with_api(clean_story, voice)
        
        # Convert audio to base64 for JSON response
        audio_files_b64 = []
        for i, audio_bytes in enumerate(audio_bytes_list):
            if audio_bytes:  # Only process non-empty audio
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                audio_files_b64.append(audio_b64)
        
        # Ensure we have at least one audio file (fallback if needed)
        if not audio_files_b64:
            fallback_audio = create_fallback_audio()
            if fallback_audio:
                audio_b64 = base64.b64encode(fallback_audio).decode('utf-8')
                audio_files_b64.append(audio_b64)
        
        # Calculate total processing time
        end_time = datetime.now()
        processing_time = str(end_time - start_time)
        
        return StoryResponse(
            audio_files=audio_files_b64,
            text=clean_story,
            event=event,
            processing_time=processing_time
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in generate_story_from_image: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Processing error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API status.
    
    Returns:
        Dictionary with health status and timestamp
    """
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """
    Root endpoint providing API information and available endpoints.
    
    Returns:
        Dictionary with API metadata and endpoint descriptions
    """
    return {
        "message": "Langate Story Generator API",
        "version": "1.0.0",
        "description": "Generate surreal stories from images with AI-powered narration",
        "endpoints": {
            "POST /generate-story": "Generate story from uploaded image",
            "GET /health": "Health check endpoint",
            "GET /docs": "Interactive API documentation",
            "GET /": "This information endpoint"
        }
    }

if __name__ == "__main__":
    # Run the FastAPI application
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)