import math
import random
import time
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import torch
import os
import pywt
import soundfile as sf

import moondream as moonmd
from openai import OpenAI
from kokoro import KPipeline
from vfp import Processor

moonModel = moonmd.vl(model="moondream-2b-int8.mf")
pipeline = KPipeline(lang_code='a')  # American English


def save_frames(processor, frame, frame_no, pos_msec):
    ts = datetime.utcfromtimestamp(pos_msec / 1000.0).time().strftime("%H%M%S.%f")
    cv2.imwrite(os.path.join(processor.params.output_dir, f"{ts}.jpg"), frame)


def blur_detect(img, threshold):
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    M, N = Y.shape
    Y = Y[0:int(M / 16) * 16, 0:int(N / 16) * 16]

    LL1, (LH1, HL1, HH1) = pywt.dwt2(Y, 'haar')
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, 'haar')
    LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, 'haar')

    E1 = np.sqrt(np.power(LH1, 2) + np.power(HL1, 2) + np.power(HH1, 2))
    E2 = np.sqrt(np.power(LH2, 2) + np.power(HL2, 2) + np.power(HH2, 2))
    E3 = np.sqrt(np.power(LH3, 2) + np.power(HL3, 2) + np.power(HH3, 2))

    def max_blocks(E, block_size):
        M, N = E.shape
        return [np.max(E[i:i + block_size, j:j + block_size])
                for i in range(0, M - block_size + 1, block_size)
                for j in range(0, N - block_size + 1, block_size)]

    Emax1 = np.array(max_blocks(E1, 8))
    Emax2 = np.array(max_blocks(E2, 4))
    Emax3 = np.array(max_blocks(E3, 2))

    EdgePoint = (Emax1 > threshold) | (Emax2 > threshold) | (Emax3 > threshold)
    RGstructure = (Emax1 < Emax2) & (Emax2 < Emax3)
    RSstructure = (Emax2 > Emax1) & (Emax2 > Emax3)

    BlurC = ((RGstructure | RSstructure) & (Emax1 < threshold)).astype(int)
    BlurExtent = np.sum(BlurC) / (np.sum(RGstructure) + np.sum(RSstructure) + 1e-9)

    return np.sum(RGstructure) / np.sum(EdgePoint), BlurExtent


def find_images(input_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                yield os.path.join(root, file)


def parse_model_output(output):
    return {
        str(i): line.split(". ", 1)[-1]
        for i, line in enumerate(output.strip().split("\n"), start=1)
    }


def generate_event(photo_obj):
    subjects = ["Strange amphibian", "major", "not so secret disposal company", "crazy duck", "very normal alien"]
    verbs = ["jumps over", "solves", "paints", "explores", "repairs", "builds", "eats", "boils"]
    adjectives = ["lazy", "mysterious", "vibrant", "ancient", "futuristic", "dark"]
    return f"A {random.choice(subjects)} {random.choice(verbs)} a {random.choice(adjectives)} {random.choice(list(photo_obj.values()))}."


def generate_prompt(event, weather, calendar, length, gemma, openai):
    dynamic_prompt = f"Setting: Langate, {calendar.month}/{calendar.day}, {weather}. Event: {event}. Create a {length}-word real-time report on this event. "
    base = """
        Create a story in present tense like it's being told by a radio community announcement host who's in the town of Langate. Act calm, and largely unbothered by supernatural happenings.
        Report in present tense on today's {month}/{day} terrifying or absurd events in a dry, eerie tone laced with dark humor.
    """.format(month=calendar.month, day=calendar.day)

    if gemma:
        return f"<start_of_turn>user\n{base}{dynamic_prompt}\n<end_of_turn>\n<start_of_turn>model"
    elif openai:
        return dynamic_prompt
    else:
        return f"{base}{dynamic_prompt}"


def text_for_table(text):
    return ' '.join(line.strip() for line in text.splitlines() if line.strip())


async def run_pipeline(video_path, weather="foggy", length=200, gemma=False, openai=True, voice="af_heart"):
    start_total_time = time.time()
    p = Processor(nth_frame=10, max_frames=2000, process_frame=save_frames, verbose=True)
    p.params.output_dir = "videos/keyframes"
    p.process(video_file=video_path)

    threshold = 35
    best_result = 1.0
    best_result_path = None

    for input_path in find_images(p.params.output_dir):
        try:
            I = cv2.imread(input_path)
            _, blurext = blur_detect(I, threshold)
            if blurext < best_result:
                best_result = blurext
                best_result_path = input_path
        except Exception:
            continue

    if not best_result_path:
        return {"error": "No suitable frame found."}

    image = Image.open(best_result_path)
    image = image.resize((image.width // 10, image.height // 10), Image.LANCZOS)
    encoded_image = moonModel.encode_image(image)

    answer = moonModel.query(encoded_image, "list three different elements of the image in order of distance")["answer"]
    parsed_variables = parse_model_output(answer)

    event = generate_event(parsed_variables)
    prompt = generate_prompt(event, weather, datetime.now(), length, gemma, openai)

    client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a radio community announcement host in Langate."},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
        model='qwen2.5:7b',
    )

    clean_result = text_for_table(chat_completion.choices[0].message.content)
    audio_files = []

    for i, (_, _, audio) in enumerate(pipeline(clean_result, voice=voice, speed=1, split_pattern=r'\n+')):
        file_path = f"output_{i}.wav"
        sf.write(file_path, audio, 24000)
        audio_files.append(file_path)

    total_time = time.time() - start_total_time
    return {
        "text": clean_result,
        "audio_files": audio_files,
        "event": event,
        "time_taken": f"{int(total_time // 60)}:{int(total_time % 60):02d}"
    }
