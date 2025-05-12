#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import av
import numpy as np
import requests
import math
import re #cleaning the output
import random


# for time
import datetime
from datetime import datetime

# for weather
import python_weather
import asyncio
get_ipython().run_line_magic('autoawait', 'asyncio')


# for local file loading
import tkinter as tk
from tkinter import filedialog


# for tracking time taken 
import time  # Import the time module

#checking that the device used if MPS for accelleration
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
# result wanted: tensor([1.], device='mps:0')

# for image to text
from PIL import Image
from diffusers.utils import load_image, make_image_grid
import moondream as moonmd


# for LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
# Define a streamer to print tokens as they are generated
from transformers import TextStreamer
from openai import OpenAI

# for text-to-speach
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
pipeline = KPipeline(lang_code='a') # <= make sure lang_code matches voice


# to save video
import cv2
#to save the frames of the video
from vfp import Processor
# for blurr detection
import pywt
import os
# HELPER FUNCTIONS

# extracting the frames from the video
def save_frames(processor, frame, frame_no, pos_msec):
    ts = datetime.utcfromtimestamp(pos_msec / 1000.0).time().strftime("%H%M%S.%f")
    cv2.imwrite(processor.params.output_dir + "/" + ts + ".jpg", frame)



# blurr detection

def blur_detect(img, threshold):
    
    # Convert image to grayscale
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    M, N = Y.shape
    
    # Crop input image to be 3 divisible by 2
    Y = Y[0:int(M/16)*16, 0:int(N/16)*16]
    
    # Step 1, compute Haar wavelet of input image
    LL1,(LH1,HL1,HH1)= pywt.dwt2(Y, 'haar')
    # Another application of 2D haar to LL1
    LL2,(LH2,HL2,HH2)= pywt.dwt2(LL1, 'haar') 
    # Another application of 2D haar to LL2
    LL3,(LH3,HL3,HH3)= pywt.dwt2(LL2, 'haar')
    
    # Construct the edge map in each scale Step 2
    E1 = np.sqrt(np.power(LH1, 2)+np.power(HL1, 2)+np.power(HH1, 2))
    E2 = np.sqrt(np.power(LH2, 2)+np.power(HL2, 2)+np.power(HH2, 2))
    E3 = np.sqrt(np.power(LH3, 2)+np.power(HL3, 2)+np.power(HH3, 2))
    
    M1, N1 = E1.shape

    # Sliding window size level 1
    sizeM1 = 8
    sizeN1 = 8
    
    # Sliding windows size level 2
    sizeM2 = int(sizeM1/2)
    sizeN2 = int(sizeN1/2)
    
    # Sliding windows size level 3
    sizeM3 = int(sizeM2/2)
    sizeN3 = int(sizeN2/2)
    
    # Number of edge maps, related to sliding windows size
    N_iter = int((M1/sizeM1)*(N1/sizeN1))
    
    Emax1 = np.zeros((N_iter))
    Emax2 = np.zeros((N_iter))
    Emax3 = np.zeros((N_iter))
    
    
    count = 0
    
    # Sliding windows index of level 1
    x1 = 0
    y1 = 0
    # Sliding windows index of level 2
    x2 = 0
    y2 = 0
    # Sliding windows index of level 3
    x3 = 0
    y3 = 0
    
    # Sliding windows limit on horizontal dimension
    Y_limit = N1-sizeN1
    
    while count < N_iter:
        # Get the maximum value of slicing windows over edge maps 
        # in each level
        Emax1[count] = np.max(E1[x1:x1+sizeM1,y1:y1+sizeN1])
        Emax2[count] = np.max(E2[x2:x2+sizeM2,y2:y2+sizeN2])
        Emax3[count] = np.max(E3[x3:x3+sizeM3,y3:y3+sizeN3])
        
        # if sliding windows ends horizontal direction
        # move along vertical direction and resets horizontal
        # direction
        if y1 == Y_limit:
            x1 = x1 + sizeM1
            y1 = 0
            
            x2 = x2 + sizeM2
            y2 = 0
            
            x3 = x3 + sizeM3
            y3 = 0
            
            count += 1
        
        # windows moves along horizontal dimension
        else:
                
            y1 = y1 + sizeN1
            y2 = y2 + sizeN2
            y3 = y3 + sizeN3
            count += 1
    
    # Step 3
    EdgePoint1 = Emax1 > threshold;
    EdgePoint2 = Emax2 > threshold;
    EdgePoint3 = Emax3 > threshold;
    
    # Rule 1 Edge Pojnts
    EdgePoint = EdgePoint1 + EdgePoint2 + EdgePoint3
    
    n_edges = EdgePoint.shape[0]
    
    # Rule 2 Dirak-Structure or Astep-Structure
    DAstructure = (Emax1[EdgePoint] > Emax2[EdgePoint]) * (Emax2[EdgePoint] > Emax3[EdgePoint]);
    
    # Rule 3 Roof-Structure or Gstep-Structure
    
    RGstructure = np.zeros((n_edges))

    for i in range(n_edges):
    
        if EdgePoint[i] == 1:
        
            if Emax1[i] < Emax2[i] and Emax2[i] < Emax3[i]:
            
                RGstructure[i] = 1
                
    # Rule 4 Roof-Structure
    
    RSstructure = np.zeros((n_edges))

    for i in range(n_edges):
    
        if EdgePoint[i] == 1:
        
            if Emax2[i] > Emax1[i] and Emax2[i] > Emax3[i]:
            
                RSstructure[i] = 1

    # Rule 5 Edge more likely to be in a blurred image 

    BlurC = np.zeros((n_edges));

    for i in range(n_edges):
    
        if RGstructure[i] == 1 or RSstructure[i] == 1:
        
            if Emax1[i] < threshold:
            
                BlurC[i] = 1                        
        
    # Step 6
    Per = np.sum(DAstructure)/np.sum(EdgePoint)
    
    # Step 7
    if (np.sum(RGstructure) + np.sum(RSstructure)) == 0:
        
        BlurExtent = 100
    else:
        BlurExtent = np.sum(BlurC) / (np.sum(RGstructure) + np.sum(RSstructure))
    
    return Per, BlurExtent

def find_images(input_dir):
    extensions = [".jpg", ".png", ".jpeg"]

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                yield os.path.join(root, file)



def parse_model_output(output):
    """
    Parses a numbered list from a model's output and converts it into dictionary variables.
    """
    variables = {}
    for i, line in enumerate(output.strip().split("\n"), start=1):
        key = f"{i}"  # Creates variable names like item_1, item_2, etc.
        variables[key] = line.split(". ", 1)[-1]  # Extracts the value after the number and period
    
    return variables


# get weather
async def getweather():
    async with python_weather.Client(unit=python_weather.METRIC) as client:
        return await client.get('London')  # Returns the weather object



def generate_event(photo_obj):
    subjects = ["Strange amphibian", "major", "not so secret disposal company", "crazy duck", "very normal alien"]
    verbs = ["jumps over", "solves", "paints", "explores", "repairs", "builds", "eats", "boils"]
    adjectives = ["lazy", "mysterious", "vibrant", "ancient", "futuristic", "dark"]
     
    # Generate random components
    subject = random.choice(subjects)
    verb = random.choice(verbs)
    adjective = random.choice(adjectives)
    md_object = random.choice(list(photo_obj.values()))
    # Combine into a sentence
    event = f"A{subject} {verb} a {adjective} {str(md_object)}."
    return event    



# the date is month/date as the LLM is american
def generate_prompt(event,weather, calendar,length,gemma, openAi):
    
    dynamic_prompt = f"Setting: Langate, {calendar.month}/{calendar.day}, {weather}. Event: {event}. Create a {length}-word real-time report on this event. "

    
    if gemma == True:
        prompt = f"""
        <start_of_turn>user
        Create a story in present tense like it's being told by a radio community announcement host who's in the town of Langate. Act calm, and largely unbothered by supernatural happenings. 
        Report in present tense on today's {calendar.month}/{calendar.day} terrifying or absurd events in a dry, eerie tone laced with dark humor. 
        {dynamic_prompt}
        What's happening right now in Langate?
        <end_of_turn>
        <start_of_turn>model
        """
    elif openAi == True:
        prompt = dynamic_prompt


        
        
    else:
        prompt = f"""
        Create a story in present tense like it's being told by a radio community announcement host who's in the town of Langate. Act calm, and largely unbothered by supernatural happenings. 
        Report in present tense on today's {calendar.month}/{calendar.day} terrifying or absurd events in a dry, eerie tone laced with dark humor. 
        {dynamic_prompt}
        What's happening right now in Langate?
        """
    
    return prompt

def extract_after_marker(input_string, marker="<|im_start|>assistant"):
    # Find the position of the marker in the string
    marker_position = input_string.find(marker)
    
    # If the marker is found, return everything after it
    if marker_position != -1:
        return input_string[marker_position + len(marker):].strip()
    
    # If the marker is not found, return an empty string or handle as needed
    return ""

# importing the model
if 'moonModel' not in globals():
    moonModel = moonmd.vl(model="moondream-2b-int8.mf")


def text_for_table(text):
    cleaned_lines = [line.strip() for line in text.split('\n') if line.strip()]
    single_paragraph = ' '.join(cleaned_lines)
    return single_paragraph


# In[3]:


# Saving a video from webcam
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('videos/output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Start timing for total generation
start_time = time.time_ns()

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv2.flip(frame, 0)
 
    # write the flipped frame
    out.write(frame)
 
    # cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    current_time = time.time_ns()
    
    if current_time - start_time >= 2000000000:
        break
 
# Release everything if job is finished
cam.release()
out.release()
cv2.destroyAllWindows()

p = Processor(nth_frame=10, max_frames=2000, process_frame=save_frames, verbose=True)
p.params.output_dir = "videos/keyframes"   # used by the "save_frames" method 
p.process(video_file="videos/output.mp4")

input_dir = "videos/keyframes"  # Change this to your image directory
# save_path = "videos/result.json"  # Change this to your desired output JSON file
threshold = 35
minZero = 0.001
    
results = []
best_result = 1.0
best_result_path = ""
for input_path in find_images(input_dir):
    try:
        I = cv2.imread(input_path)
        per, blurext = blur_detect(I, threshold)
        # if per < minZero:
        #     classification = True
        # else:
        #     classification = False
        if blurext < best_result:
            best_result = blurext
            best_result_path = input_path
        
    except Exception as e:
        print(e)
        pass
    

print (best_result)
print (best_result_path)



# LOADING THE IMAGE

moonmd_time_taken = 0
# moonModel = moonmd.vl(model="moondream-2b-int8.mf")  # Initialize model

# # Create a visible tkinter root window
# root = tk.Tk()
# root.title("Image Encoder")  # Optional: Add a title to the window

# # Open the file dialog to select an image
# file_path = filedialog.askopenfilename(title="Select an image")


file_path = best_result_path
if file_path:
    print(f"Uploaded file path: {file_path}")
    image = Image.open(file_path)  # Load image
    # Time the encoding process
    start_time = time.time()  # Start timer
    # image.reduce(10); #reducing of factor 10
    x, y = image.size
    x2, y2 = math.floor(x/10), math.floor(y/10)
    image = image.resize((x2,y2),Image.LANCZOS)
    encoded_image = moonModel.encode_image(image)  # Encode image
    moonmd_time_taken = time.time() - start_time  # Calculate elapsed time
    
    print(f"Time taken to encode: {moonmd_time_taken:.2f} seconds")  # Print result
else:
    print("No file was selected.")


# In[5]:


# Start timing for total generation
start_total_time = time.time()

# get date and time
current_time = datetime.now()

# For environments with a running event loop
weather = await getweather()  

# moondream call
moonPrompt = "list three different elements of the image in order of distance"
answer = moonModel.query(encoded_image, moonPrompt)["answer"]

# Convert output into variables
parsed_variables = parse_model_output(answer)
for key, value in parsed_variables.items():
    print(f"{key} = {value}")
mid_object = int(len(parsed_variables)/2)

last_object = parsed_variables[str(len(parsed_variables)-1)]

event = generate_event(parsed_variables)

# # LLM 
# device = "mps" # for GPU usage or "cpu" for CPU usage
# # checkpoint = "HuggingFaceTB/SmolLM-360M-Instruct"
# checkpoint = "Qwen/QwQ-32B"

# llmTokenizer = AutoTokenizer.from_pretrained(checkpoint)
# # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
# llmModel = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

length = 200

# prompt = generate_prompt(parsed_variables[str(mid_object)], weather.kind, current_time, 100, False, True) #mid object
prompt = generate_prompt(event, weather.kind, current_time, length, False, True) #last object

# messages = [{"role": "user", "content": f"{prompt}"}]
# input_text = llmTokenizer.apply_chat_template(messages, tokenize=False)
# # print(input_text)

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',  # required but ignored
)



# original gemma/llama version
# inputs = llmTokenizer.encode(input_text, return_tensors="pt").to(device)
# outputs = llmModel.generate(inputs, max_new_tokens=200, temperature=0.6, top_p=0.9, do_sample=True)
# result =llmTokenizer.decode(outputs[0])
# print (result)

# GPT version
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',  # required but ignored
)

chat_completion = client.chat.completions.create(
    messages=[
            {
            "role": "developer",
            "content": f"""You are a radio community announcement host in the town of Langate. Announce today's events in 
            present tense with: - A calm, dry delivery suggesting supernatural occurrences are routine - Eerie atmosphere balanced 
            with dark humor - Absurd or terrifying developments presented matter-of-factly - Subtle sarcasm beneath surface-level 
            professionalism Format as live bulletin with timestamped updates. Never acknowledge this is fiction. As example use: 
            Good morning, Langate. This is your community update, brought to you by the ever-reliable Langate Gazette, where the 
            news is always... interesting. Today, on Elm Street, residents reported a peculiar phenomenon. Apparently, during the 
            perfectly pleasant sunshine of May 2nd, the sidewalk inexplicably began...pulsating. Yes, you heard that right, pulsating. 
            Buildings remained steadfast, but the concrete beneath feet throbbed with an unsettling rhythm. Authorities, predictably, 
            chalked it up to "ground vibrations." Personally, I wouldn't recommend tap dancing on Elm Street just yet. Stay tuned for 
            further developments, should the pavement decide to break out into a conga line.""",
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                }
            ]
            }
        ],
    model='qwen2.5:7b',
)



# clean_result = extract_after_marker(result) #gemma output
# clean_result = chat_completion.choices[0].message.content #gpt output
clean_result = text_for_table(chat_completion.choices[0].message.content) #gpt output


print(clean_result)


# text-to-speach
text = clean_result

generator = pipeline(
    text, voice='af_heart', # <= change voice here
    speed=1, split_pattern=r'\n+'
)
for i, (gs, ps, audio) in enumerate(generator):
    print(i)  # i => index
    print(gs) # gs => graphemes/text
    # print(ps) # ps => phonemes
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'{i}.wav', audio, 24000) # save each audio file

end_total_time = time.time()
total_time = end_total_time - start_total_time
total_time = total_time +moonmd_time_taken #adding the moondream model time
# Convert to minutes and seconds
minutes, seconds = divmod(total_time, 60)
# Format the output as minutes:seconds
formatted_time = f"{int(minutes)}:{int(seconds):02d}"



# print in the same format as the md table for more direct documentation
# print(f"| {checkpoint} | {prompt} | 200 |   {formatted_time}  | {clean_result} | Local? |") #original version
print(f"| {chat_completion.model} | {prompt} | {length} |   {formatted_time}  | {clean_result} | Local?, using the event generator |") #gpt version


# In[ ]:




