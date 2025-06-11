# STROIES EVERYWHERE
## Rachele Guzzon

Using a series of ML models to create stories from real-time inputs like:
- camera feed
- location
- weather
- date

# Event generator
### goals:
- aid model's consistency choosing from set characters
- generate more absurd events
### structure
random selection of one of each of the following cathegories:
- subject
- **object (one of the moondream's outputs)**
- adjective
- verb

### Partial Results (full in doc notebook):


| Model Name  | Prompt Length (Tokens) | Tokens Generated |  Time Taken (all models combined) (Minutes:Seconds) | Results |Notes      |
|-------------|------------------------|------------------|-----------------------|---------|-----------|
| qwen2.5:7b | Setting: Langate, 3/16, Partly Cloudy. Event: A crane in the background.. Create a 200-word real-time report on this event.  | 200 |   1:08  | Real-Time Report:Date and Time: March 16, 2023, 1:45 PM Location: Langate. Current Weather Conditions: Partly Cloudy, with an atmospheric temperature of 18°C (64°F) and a gentle northwesterly breeze. Event Description: At the construction site in Langate, a towering crane has been positioned prominently in the background. The sun is filtering through the clouds, casting a golden glow upon the structure that rises from the ground like a modernist skyscraper. Workers are bustling around, ensuring that each task is carried out flawlessly under the watchful eye of experienced professionals. The crane stands as a symbol of progress and advancement, its jib stretching skyward to connect with the building's skeletal framework. Nearby cranes offer a harmonious backdrop, creating an impressionist painting of industrious activity and human ingenuity in action. The air is filled with the hum of machinery and voices working together in perfect rhythm. Local residents are stopping to gaze at the scene, intrigued by this grand spectacle of construction and architectural marvel. Photographers are setting up their equipment, capturing moments that will become memories for years to come. The Langate community can take pride in witnessing firsthand how their city is growing and evolving before their eyes. | Local? |
| qwen2.5:7b | Setting: Langate, 3/16, Partly Cloudy. Event: AStrange amphibian builds a futuristic A canal with a lock... Create a 200-word real-time report on this event.  | 200 |   1:23  | Real-Time Report: Strange Amphibian Canal Construction Observed in Langate, March 16 Date: March 16, 2023. Location: Langate, Cloudy but partly sunny conditions with occasional showers  . In a stunning display of unexplained natural phenomena, an amphibian species previously undocumented by local or global scientific communities has been observed constructing what appears to be a high-tech canal system capable of lock operations. This extraordinary discovery was made in the early morning hours at the edge of Langate's expansive marshlands. The alien-like creatures work tirelessly, using their intricate leg and tail structures to dig canals with precision that resembles advanced engineering techniques. Most remarkable is the presence of what appears to be a rudimentary canal lock system, something typically associated with human infrastructure developments rather than natural amphibian behavior. Local authorities and researchers from the nearby city science institute have been tasked with investigating this unprecedented event. Speculation ranges widely—from a previously unknown species exhibiting unusually complex behaviors due to an as-yet unexplained evolutionary leap—hypotheses that suggest advanced intelligence beyond current scientific understanding in the animal kingdom. Eyewitnesses report that these creatures seem to communicate intricately during construction and operate in groups, moving equipment and materials with astonishing efficiency. Further analysis is needed to understand how this amphibian species has acquired such abilities and why it chose this specific location for its endeavors. This unique event underscores the need for more comprehensive research into the natural world and challenges our current understanding of biological capabilities. Science and local communities across Langate are keeping a close eye on developments to better comprehend this extraordinary occurrence. For updates and further information, stay tuned. | Local?, using the event generator |

# API
A FastAPI service that transforms an image into a short eerie story and an accompanying audio clip, inspired by the town of **Langate**.
Deployed through Render

---

##  Features

*  Analyzes the image and extracts scene elements
*  Generates a surreal, darkly humorous story based on the weather, date, and image
*  Converts the story into audio using Hugging Face TTS
* Uses large language models (LLMs) and vision models for end-to-end story synthesis

---

## API Endpoints

### `POST /generate-story`

Generate a Langate story and audio clip from an image.

####  Request (multipart/form-data)

**Parameters (query or form):**

* `weather`: (`string`) default=`foggy` –  weather placeholder
* `length`: (`int`) default=`200` – approximate story length in words
* `voice`: (`string`) default=`af_heart` – voice identifier (currently unused placeholder)

**Body:**

* `file`: (`image/*`) The uploaded image

####  Response: `200 OK`

```json
{
  "audio_files": ["<base64_encoded_audio>"],
  "text": "Story goes here...",
  "event": "A very normal alien explores a vibrant tree.",
  "processing_time": "0:00:06.124321"
}
```

---


### `GET /`

Root info.

```json
{
  "message": "Langate Story Generator API",
  "version": "1.0.0",
  "endpoints": {
    "POST /generate-story": "Generate story from image",
    "GET /health": "Health check",
    "GET /docs": "API documentation"
  }
}
```

---

## Setup & Run

###  Requirements

* Python 3.9+
* [Hugging Face API Key](https://huggingface.co/settings/tokens)

###  Installation

```bash
pip install -r requirements.txt
```

###  Run the server

```bash
uvicorn main:app --reload
```

Make sure to set your Hugging Face token in your environment:

```bash
export HUGGINGFACE_API_KEY=your_token_here
```

---

##  Response Example (Decoded)

```json
{
  "text": "In Langate today, a strange amphibian paints a mysterious building...",
  "event": "A strange amphibian paints a mysterious building.",
  "audio_files": ["UklGRu...<truncated base64 audio>..."]
}
```

To convert the base64 audio to playable `.wav`:

```python
import base64
with open("output.wav", "wb") as f:
    f.write(base64.b64decode(audio_files[0]))
```



