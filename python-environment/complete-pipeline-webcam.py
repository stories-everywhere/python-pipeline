#!/usr/bin/env python
# coding: utf-8

import sys
import av
import numpy as np
import requests
import math
import re  # cleaning the output
import random
import datetime
from datetime import datetime
import python_weather
import asyncio
import tkinter as tk
from tkinter import filedialog
import time
import torch
from PIL import Image
from diffusers.utils import load_image, make_image_grid
import moondream as moonmd
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
from transformers import TextStreamer
from openai import OpenAI
from kokoro import KPipeline
import soundfile as sf
import cv2
from vfp import Processor
import pywt
import os

# Initialize moondream model globally
moonModel = moonmd.vl(model="moondream-2b-int8.mf")
pipeline = KPipeline(lang_code='a')  # American English

# HELPER FUNCTIONS

def save_frames(processor, frame, frame_no, pos_msec):
    ts = datetime.utcfromtimestamp(pos_msec / 1000.0).time().strftime("%H%M%S.%f")
    cv2.imwrite(processor.params.output_dir + "/" + ts + ".jpg", frame)

def blur_detect(img, threshold):
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    M, N = Y.shape
    Y = Y[0:int(M/16)*16, 0:int(N/16)*16]
    
    LL1, (LH1, HL1, HH1) = pywt.dwt2(Y, 'haar')
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, 'haar') 
    LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, 'haar')
    
    E1 = np.sqrt(np.power(LH1, 2)+np.power(HL1, 2)+np.power(HH1, 2))
    E2 = np.sqrt(np.power(LH2, 2)+np.power(HL2, 2)+np.power(HH2, 2))
    E3 = np.sqrt(np.power(LH3, 2)+np.power(HL3, 2)+np.power(HH3, 2))
    
    M1, N1 = E1.shape
    sizeM1 = 8
    sizeN1 = 8
    sizeM2 = int(sizeM1/2)
    sizeN2 = int(sizeN1/2)
    sizeM3 = int(sizeM2/2)
    sizeN3 = int(sizeN2/2)
    N_iter = int((M1/sizeM1)*(N1/sizeN1))
    
    Emax1 = np.zeros((N_iter))
    Emax2 = np.zeros((N_iter))
    Emax3 = np.zeros((N_iter))
    
    count = 0
    x1, y1 = 0, 0
    x2, y2 = 0, 0
    x3, y3 = 0, 0
    Y_limit = N1-sizeN1
    
    while count < N_iter:
        Emax1[count] = np.max(E1[x1:x1+sizeM1,y1:y1+sizeN1])
        Emax2[count] = np.max(E2[x2:x2+sizeM2,y2:y2+sizeN2])
        Emax3[count] = np.max(E3[x3:x3+sizeM3,y3:y3+sizeN3])
        
        if y1 == Y_limit:
            x1 += sizeM1
            y1 = 0
            x2 += sizeM2
            y2 = 0
            x3 += sizeM3
            y3 = 0
            count += 1
        else:
            y1 += sizeN1
            y2 += sizeN2
            y3 += sizeN3
            count += 1
    
    EdgePoint1 = Emax1 > threshold
    EdgePoint2 = Emax2 > threshold
    EdgePoint3 = Emax3 > threshold
    EdgePoint = EdgePoint1 + EdgePoint2 + EdgePoint3
    n_edges = EdgePoint.shape[0]
    
    DAstructure = (Emax1[EdgePoint] > Emax2[EdgePoint]) * (Emax2[EdgePoint] > Emax3[EdgePoint])
    RGstructure = np.zeros((n_edges))

    for i in range(n_edges):
        if EdgePoint[i] == 1:
            if Emax1[i] < Emax2[i] and Emax2[i] < Emax3[i]:
                RGstructure[i] = 1
                
    RSstructure = np.zeros((n_edges))
    for i in range(n_edges):
        if EdgePoint[i] == 1:
            if Emax2[i] > Emax1[i] and Emax2[i] > Emax3[i]:
                RSstructure[i] = 1

    BlurC = np.zeros((n_edges))
    for i in range(n_edges):
        if RGstructure[i] == 1 or RSstructure[i] == 1:
            if Emax1[i] < threshold:
                BlurC[i] = 1                        
    
    Per = np.sum(DAstructure)/np.sum(EdgePoint)
    
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
    variables = {}
    for i, line in enumerate(output.strip().split("\n"), start=1):
        key = f"{i}"
        variables[key] = line.split(". ", 1)[-1]
    return variables

async def getweather():
    async with python_weather.Client(unit=python_weather.METRIC) as client:
        return await client.get('London')

def generate_event(photo_obj):
    subjects = ["Strange amphibian", "major", "not so secret disposal company", "crazy duck", "very normal alien"]
    verbs = ["jumps over", "solves", "paints", "explores", "repairs", "builds", "eats", "boils"]
    adjectives = ["lazy", "mysterious", "vibrant", "ancient", "futuristic", "dark"]
     
    subject = random.choice(subjects)
    verb = random.choice(verbs)
    adjective = random.choice(adjectives)
    md_object = random.choice(list(photo_obj.values()))
    event = f"A {subject} {verb} a {adjective} {str(md_object)}."
    return event    

def generate_prompt(event, weather, calendar, length, gemma, openAi):
    dynamic_prompt = f"Setting: Langate, {calendar.month}/{calendar.day}, {weather}. Event: {event}. Create a {length}-word real-time report on this event. "
    
    if gemma:
        prompt = f"""
        <start_of_turn>user
        Create a story in present tense like it's being told by a radio community announcement host who's in the town of Langate. Act calm, and largely unbothered by supernatural happenings. 
        Report in present tense on today's {calendar.month}/{calendar.day} terrifying or absurd events in a dry, eerie tone laced with dark humor. 
        {dynamic_prompt}
        What's happening right now in Langate?
        <end_of_turn>
        <start_of_turn>model
        """
    elif openAi:
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
    marker_position = input_string.find(marker)
    if marker_position != -1:
        return input_string[marker_position + len(marker):].strip()
    return ""

def text_for_table(text):
    cleaned_lines = [line.strip() for line in text.split('\n') if line.strip()]
    single_paragraph = ' '.join(cleaned_lines)
    return single_paragraph

async def main():
    # Check for MPS acceleration
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")

    # Video capture
    cam = cv2.VideoCapture(0)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('videos/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    start_time = time.time_ns()

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv2.flip(frame, 0)
        out.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break
        current_time = time.time_ns()
        if current_time - start_time >= 2000000000:
            break

    cam.release()
    out.release()
    cv2.destroyAllWindows()

    # Process frames
    p = Processor(nth_frame=10, max_frames=2000, process_frame=save_frames, verbose=True)
    p.params.output_dir = "videos/keyframes"
    p.process(video_file="videos/output.mp4")

    # Find best frame
    input_dir = "videos/keyframes"
    threshold = 35
    best_result = 1.0
    best_result_path = ""
    
    for input_path in find_images(input_dir):
        try:
            I = cv2.imread(input_path)
            per, blurext = blur_detect(I, threshold)
            if blurext < best_result:
                best_result = blurext
                best_result_path = input_path
        except Exception as e:
            print(e)
            pass
    
    print(best_result)
    print(best_result_path)

    # Process image
    if not best_result_path:
        print("No suitable image found")
        return

    print(f"Selected file path: {best_result_path}")
    image = Image.open(best_result_path)
    start_time = time.time()
    x, y = image.size
    x2, y2 = math.floor(x/10), math.floor(y/10)
    image = image.resize((x2,y2), Image.LANCZOS)
    encoded_image = moonModel.encode_image(image)
    moonmd_time_taken = time.time() - start_time
    print(f"Time taken to encode: {moonmd_time_taken:.2f} seconds")

    # Generate content
    start_total_time = time.time()
    current_time = datetime.now()
    weather = await getweather()
    
    moonPrompt = "list three different elements of the image in order of distance"
    answer = moonModel.query(encoded_image, moonPrompt)["answer"]
    parsed_variables = parse_model_output(answer)
    
    for key, value in parsed_variables.items():
        print(f"{key} = {value}")
    
    event = generate_event(parsed_variables)
    length = 200
    prompt = generate_prompt(event, weather.kind, current_time, length, False, True)

    # Generate text
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama',
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "developer",
                "content": """You are a radio community announcement host in the town of Langate...""",
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

    clean_result = text_for_table(chat_completion.choices[0].message.content)
    print(clean_result)

    # Text-to-speech
    text = clean_result
    generator = pipeline(
        text, voice='af_heart',
        speed=1, split_pattern=r'\n+'
    )
    
    for i, (gs, ps, audio) in enumerate(generator):
        print(i)
        print(gs)
        sf.write(f'{i}.wav', audio, 24000)

    end_total_time = time.time()
    total_time = end_total_time - start_total_time + moonmd_time_taken
    minutes, seconds = divmod(total_time, 60)
    formatted_time = f"{int(minutes)}:{int(seconds):02d}"

    print(f"| {chat_completion.model} | {prompt} | {length} | {formatted_time} | {clean_result} | Local?, using the event generator |")

if __name__ == "__main__":
    asyncio.run(main())