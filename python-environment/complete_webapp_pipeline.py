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
    date: str  = "00:00"

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
        "crazy duck", "very normal alien", 
        "quantum cat", "sentient vending machine", 
        "robot barista", "suspicious librarian", 
        "moon janitor", "bio-luminescent intern", 
        "tiny chaos deity", "grumpy cloud", "escaped simulation"
    ]

    verbs = [
        "jumps over", "solves", "paints", "explores", 
        "repairs", "builds", "eats", "boils", 
        "teleports", "invents", "whispers to", "disguises as", 
        "encrypts", "deconstructs", "haunts", "accidentally befriends", 
        "trades with", "measures", "reverses", "downloads from", 
        "conducts", "maps", "photocopies", "upgrades"
    ]

    adjectives = [
        "lazy", "mysterious", "vibrant", "ancient", 
        "futuristic", "noisy", "invisible", 
        "radioactive", "cursed", "electric", "sticky", 
        "floating", "bored", "recycled", "unstable", 
        "delusional", "glowing"
    ]

    
    # # Select a random element from the photo or use default
    # element = random.choice(list(photo_elements.values())) if photo_elements else "object"
    
    # return f"A {random.choice(subjects)} {random.choice(verbs)} a {random.choice(adjectives)} {element}."

    #include all elements in the event    
    elements = list(photo_elements.values())

    if not elements:
        return "No recognizable elements were found in the image."

    if len(elements) == 1:
        description = elements[0]
    elif len(elements) == 2:
        description = f"{elements[0]} and {elements[1]}"
    else:
        description = ", ".join(elements[:-1]) + f", and {elements[-1]}"
    # adjective of desccription has been erased to keep more of the real-life descriptors
    return (
        f"The {random.choice(subjects)} {random.choice(verbs)} "
        f" {description}."
    )
    # return f"The {random.choice(subjects)} {random.choice(verbs)}  {random.choice(adjectives)} {photo_elements['1']}, {photo_elements['2']} and {photo_elements['3']}."

def generate_prompt(event: str, weather: str, date: str, length: int) -> str:
    """
    Generate a prompt for story generation based on parameters.
    
    Args:
        event: The generated event description
        weather: Current weather condition
        date: Current time
        length: Desired story length in words
        
    Returns:
        Formatted prompt string for the AI model
    """
    # dynamic_prompt = (
    #     # f"Setting: Langate, {calendar.month}/{calendar.day}, {weather}. "
    #     f"Event: {event}. Create a {length}-word real-time report on this event. "
    # )
    
    base_prompt = """
        At {} {}. 
        Create a {}-word  report on this event without restating the above sentence but including the hour. 
    """.format(date, event, length)
    
    return f"{base_prompt.strip()}"

def clean_text(text: str) -> str:
    """
    Clean and format text for processing.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text with proper spacing
    """
    return ' '.join(line.strip() for line in text.splitlines() if line.strip())

def split_text_into_blocks(text, max_length=500):
    """
    Split text into blocks of maximum specified length, with intelligent punctuation-based cutting.
    
    Args:
        text (str): The input text to split
        max_length (int): Maximum length of each block (default: 500)
    
    Returns:
        list: List of text blocks, each ending with punctuation or cut at max length
        
    The function tries to cut at punctuation marks in this order:
    1. Full stop (.)
    2. Semicolon (;)
    3. Comma (,)
    4. Colon (:)
    5. If none found, cuts at max length
    """
    if not text:
        return []
    
    blocks = []
    current_pos = 0
    
    while current_pos < len(text):
        # Find the end position for this block
        end_pos = min(current_pos + max_length, len(text))
        
        # If we've reached the end of the text, take the remaining part
        if end_pos >= len(text):
            remaining_text = text[current_pos:].strip()
            if remaining_text:
                blocks.append(remaining_text)
            break
        
        # Look for the best cut point within the max length
        block_text = text[current_pos:end_pos]
        
        # Try to find punctuation marks in order of preference
        cut_pos = -1
        cut_char = None
        
        # First try full stop
        last_period_pos = block_text.rfind('.')
        if last_period_pos != -1:
            cut_pos = last_period_pos
            cut_char = '.'
        else:
            # Try semicolon
            last_semicolon_pos = block_text.rfind(';')
            if last_semicolon_pos != -1:
                cut_pos = last_semicolon_pos
                cut_char = ';'
            else:
                # Try comma
                last_comma_pos = block_text.rfind(',')
                if last_comma_pos != -1:
                    cut_pos = last_comma_pos
                    cut_char = ','
                else:
                    # Try colon
                    last_colon_pos = block_text.rfind(':')
                    if last_colon_pos != -1:
                        cut_pos = last_colon_pos
                        cut_char = ':'
        
        if cut_pos != -1:
            # Found a suitable punctuation mark
            actual_end_pos = current_pos + cut_pos + 1
            block_text = text[current_pos:actual_end_pos]
            blocks.append(block_text.strip())
            current_pos = actual_end_pos
        else:
            # No suitable punctuation found, cut at max length
            block_text = text[current_pos:end_pos]
            blocks.append(block_text.strip())
            current_pos = end_pos
    
    return blocks

async def analyze_image_with_api(image_data: bytes) -> Dict[str, str]:
    """
    Analyze an image using MoonDream API to identify elements.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Dictionary of identified image elements
    """
    global md_client
    if not md_client:
        print("MoonDream client not available, using fallback")
        return {"1": "building", "2": "tree", "3": "sky"}

    try:
        print(f"Received image data: {len(image_data)} bytes")
        print(f"First 20 bytes: {image_data[:20]}")
        
        # Convert raw bytes to PIL Image first
        image = Image.open(io.BytesIO(image_data))
        print(f"PIL Image opened successfully: {image.format}, {image.mode}, {image.size}")
        
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            print(f"Converting from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Convert PIL Image to JPEG bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=85)
        jpeg_bytes = img_buffer.getvalue()
        print(f"Converted to JPEG: {len(jpeg_bytes)} bytes")
        md_prompt = "Identify three distinct elements in this image. For each, provide a detailed description including attributes such as color, shape, or texture. Format as: [element], [element], [element]"
        # Try different approaches based on MoonDream API expectations
        # Approach 1: Direct bytes
        try:
            print("Trying direct JPEG bytes...")
            response = md_client.query(
                jpeg_bytes, 
                md_prompt
            )
            response_content = response['answer']
            print(f"Success with direct bytes: {response_content}")
            return parse_model_output(response_content)
        except Exception as e1:
            print(f"Direct bytes failed: {e1}")
            
        # Approach 2: Base64 encoded
        try:
            print("Trying base64 encoded...")
            image_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
            response = md_client.query(
                f"data:image/jpeg;base64,{image_b64}", 
                md_prompt
            )
            response_content = response['answer']
            print(f"Success with base64: {response_content}")
            return parse_model_output(response_content)
        except Exception as e2:
            print(f"Base64 failed: {e2}")
            
        # Approach 3: PIL Image object directly
        try:
            print("Trying PIL Image object...")
            response = md_client.query(
                image, 
                md_prompt
            )
            response_content = response['answer']
            print(f"Success with PIL Image: {response_content}")
            return parse_model_output(response_content)

        except Exception as e3:
            print(f"PIL Image failed: {e3}")

    except Exception as e:
        print(f"Error analyzing image: {e}")
        import traceback
        traceback.print_exc()
        # Return fallback elements on error
        return {"building, tree, sky "}

async def generate_story_with_api(prompt: str) -> str:
    """
    Generate a story using Hugging Face text generation API.
    
    Args:
        prompt: The story generation prompt
        
    Returns:
        Generated story text
    """

    global openai_client
    if not openai_client:
        print("Open Ai [sambanova] client not available, using fallback")
        return (
            "In the misty town of Langate today, residents report unusual "
            "occurrences involving local wildlife and mysterious structures. "
            "The mayor assures everyone this is perfectly normal for a Tuesday."
        )
    
    try:
       
        # deepSeek call
        # response = openai_client.chat.completions.create(
        #     model="DeepSeek-V3-0324",
        #     messages=[
        #         # {
        #         #     "role":"system",
        #         #     "content":"You are a helpful assistant"
        #         # },
        #         {
        #             "role":"user",
        #             "content":prompt
        #         }
        #     ],
        #     temperature=0.1,
        #     top_p=0.1
        # )

        #llama call
        response = openai_client.chat.completions.create(
            model="Meta-Llama-3.1-8B-Instruct",
            messages=[{
                "role":"system",
                "content":
                    """ 
                    You are a radio announcer in the town of Langate, broadcasting in the middle of your day. You report on supernatural events as soon as you become aware of them, always stating the hour they occur. Begin each report with phrases such as "I've just been told that at hour...".
                    
                    The events may be terrifying, absurd, or both. Report them in a calm, dry, and eerie tone, as though such happenings are routine. Your delivery should carry a subtle thread of dark humor — the kind that suggests you're either slightly amused or entirely resigned to the madness of Langate.

                    After each report, sound as though you are waiting for the next unsettling message to arrive.

                    Your output should be a transcript of only spoken words intended for a text-to-speech model. Use plain text only. Do not include any special characters except for quotation marks ("), and include nothing outside of what the voice should say.

                    Do not include any non-verbal cues or stage directions such as (pause), (sigh), or sound effects. 
                    """
                },
            {
                "role":"user",
                "content":prompt
                }],
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

async def generate_audio_with_api(text_blocks: List[str], voice: str) -> List[bytes]:
    """
    Generate audio from text blocks using FAL AI text-to-speech API.
    
    Args:
        text_blocks: List of text blocks to convert to speech
        voice: Voice identifier (currently unused)
        
    Returns:
        List of audio bytes (one for each text block)
    """
    audio_results = []
    
    for i, text_block in enumerate(text_blocks):
        print(f"Processing block {i+1}/{len(text_blocks)}: {len(text_block)} chars")
        
        try:
            # Submit async request to FAL AI TTS service
            handler = await fal_client.submit_async(
                "fal-ai/chatterbox/text-to-speech",
                arguments={
                    "audio_url": "https://v3.fal.media/files/elephant/wLS77pG8fFjdqybPlKp3g_1-common_voice_en_39613299.mp3",
                    "exaggeration": 0.3,
                    "temperature": 0.7,
                    "cfg": 0.5,
                    "text": text_block
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
                print(f"Generated audio URL for block {i+1}: {audio_url}")
                
                # Fetch the audio file from the URL
                audio_bytes = await fetch_audio_from_url(audio_url)
                
                if audio_bytes:
                    audio_results.append(audio_bytes)
                else:
                    print(f"Failed to fetch audio from URL for block {i+1}, using fallback")
                    audio_results.append(create_fallback_audio())
            else:
                print(f"No audio URL found in FAL API response for block {i+1}, using fallback")
                audio_results.append(create_fallback_audio())
                 
        except Exception as e:
            print(f"Error generating audio for block {i+1}: {e}")
            audio_results.append(create_fallback_audio())
    
    return audio_results


# Helper function to use the text splitter with audio generation
async def generate_audio_from_text(text: str, voice: str , max_block_length: int = 500) -> List[bytes]:
    """
    Split text into blocks and generate audio for each block.
    
    Args:
        text: Text to convert to speech
        voice: Voice identifier
        max_block_length: Maximum length of each text block
        
    Returns:
        List of audio bytes (one for each text block)
    """
    # Split text into blocks
    text_blocks = split_text_into_blocks(text, max_block_length)
    
    if not text_blocks:
        print("No text blocks to process")
        return []
    
    print(f"Split text into {len(text_blocks)} blocks")
    
    # Generate audio for each block
    return await generate_audio_with_api(text_blocks, voice)

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
    length: int = 300,
    voice: str = "af_heart"
    date: str = "00:00"
):
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
        
        
        # Analyze image to extract elements
        photo_elements = await analyze_image_with_api(image_data)
        
        # Rest of your code remains the same...
        event = generate_event(photo_elements)
        prompt = generate_prompt(event, weather, date, length)
        print(f"Prompt: {prompt}")

        story_text = await generate_story_with_api(prompt)
        clean_story = clean_text(story_text)
        
        audio_bytes_list = await generate_audio_from_text(clean_story, voice)
        
        audio_files_b64 = []
        for i, audio_bytes in enumerate(audio_bytes_list):
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                audio_files_b64.append(audio_b64)
        
        if not audio_files_b64:
            fallback_audio = create_fallback_audio()
            if fallback_audio:
                audio_b64 = base64.b64encode(fallback_audio).decode('utf-8')
                audio_files_b64.append(audio_b64)
        
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