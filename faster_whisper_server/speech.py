
from collections.abc import Iterator
from copy import copy
from functools import partial
import os
import re
import threading
from time import sleep, time
from typing import Any, Dict, Iterator, Literal, Tuple  # noqa: UP035

import gradio as gr
from gradio import Timer
import httpx
from lager import log
from langdetect import detect
import numpy as np
from openai import OpenAI

# from pyannote.audio import Pipeline
from rich.console import Console
import soundfile as sf
from TTS.api import TTS

from faster_whisper_server.audio_task import get_audio_state, update_audio_state
from faster_whisper_server.transition import get_state, update_state

agent_base_url: str = "http://localhost:3389/v1"
agent_token: str = "mbodi-demo-1"
timeout: int = 10
TRANSCRIPTION_ENDPOINT: str = "/audio/transcriptions"
TRANSLATION_ENDPOINT: str = "/audio/translations"
TIMEOUT_SECONDS: int = 180
WEBSOCKET_URI: str = "wss://api.mbodi.ai/audio/v1/transcriptions"
CHUNK: int = 1024
CHANNELS: int = 1
RATE: int = 16000
PLACE_HOLDER: str = "Loading can take 30 seconds if a new model is selected..."
TTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
FIRST_SPEAKER: str = "Aaron Dreschner"
SECOND_SPEAKER: str = "Sofia Hellen"
FIRST_LANGUAGE: str = "en"
SECOND_LANGUAGE: str = "es"
GPU: str = "cuda:6"

SPEAKERS = [
    "Claribel Dervla",
    "Daisy Studious",
    "Gracie Wise",
    "Tammie Ema",
    "Alison Dietlinde",
    "Ana Florence",
    "Annmarie Nele",
    "Asya Anara",
    "Brenda Stern",
    "Gitta Nikolina",
    "Henriette Usha",
    "Sofia Hellen",
    "Tammy Grit",
    "Tanja Adelina",
    "Vjollca Johnnie",
    "Andrew Chipper",
    "Badr Odhiambo",
    "Dionisio Schuyler",
    "Royston Min",
    "Viktor Eka",
    "Abrahan Mack",
    "Adde Michal",
    "Baldur Sanjin",
    "Craig Gutsy",
    "Damien Black",
    "Gilberto Mathias",
    "Ilkin Urbano",
    "Kazuhiko Atallah",
    "Ludvig Milivoj",
    "Suad Qasim",
    "Torcull Diarmuid",
    "Viktor Menelaos",
    "Zacharie Aimilios",
    "Nova Hogarth",
    "Maja Ruoho",
    "Uta Obando",
    "Lidiya Szekeres",
    "Chandra MacFarland",
    "Szofi Granger",
    "Camilla Holmström",
    "Lilya Stainthorpe",
    "Zofija Kendrick",
    "Narelle Moon",
    "Barbora MacLean",
    "Alexandra Hisakawa",
    "Alma María",
    "Rosemary Okafor",
    "Ige Behringer",
    "Filip Traverse",
    "Damjan Chapman",
    "Wulf Carlevaro",
    "Aaron Dreschner",
    "Kumar Dahl",
    "Eugenio Mataracı",
    "Ferran Simen",
    "Xavier Hayasaka",
    "Luis Moray",
    "Marcos Rudaski",
]

speech_settings = {
    "speaker": FIRST_SPEAKER,
    "language": FIRST_LANGUAGE,
    "second_speaker": SECOND_SPEAKER,
    "second_language": SECOND_LANGUAGE,
    "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
    "gpu": True,
    "chunk_size": 1024,
    "chunk_duration": 0.25,
    "sr": 16000,
}
speech_state = {
    "speak_mode": "clear",
    "response": "",
    "spoken": "",
    "uncommitted": "",
    "audio_array": np.array([], dtype=np.int16),
}
lock = threading.Lock()

def get_speech_state():
    with lock:
        return speech_state.copy()
def update_speech_state(new_state):
    with lock:
        speech_state.update(new_state)

settings_lock = threading.Lock() 
def get_speech_settings():
    with settings_lock:
        return speech_settings.copy()

def update_speech_settings(new_settings):
    with settings_lock:
        speech_settings.update(new_settings)

if gr.NO_RELOAD:
    settings = get_speech_settings()
    model = settings["tts_model"]
    gpu = settings["gpu"]
    tts = TTS(model, gpu)

def speak(text: str, state: Dict, speaker: str, language: str, second_speaker: str=SPEAKERS[0], second_language: str="ru") -> Iterator[Tuple[Tuple[int, np.ndarray], Dict]]:
    """Generate and stream TTS audio using Coqui TTS with diarization."""
    state = get_state()
    print(f"SPEAK: Initial state: {state}")
    text = state["response"]
    mode = state["speak_mode"]
    sr = tts.synthesizer.output_sample_rate

    print(f"Input text: {text}")
    print(f"Initial mode: {mode}")

    if not text or (len(text.split()) < 3 and not text.endswith((".", "?", "!"))) or mode in ("clear", "wait"):
        state["speak_mode"] = "wait"
        print("Text too short or mode is clear/wait. Yielding empty array.")
        yield (sr, np.array([], dtype=np.int16)), state
        return

    # Detect language and choose speaker
    detected_language = detect(text)
    current_speaker = second_speaker if detected_language == second_language else speaker
    print(f"Detected language: {detected_language}, Selected speaker: {current_speaker}")

    # Generate full audio
    print("Generating full audio...")
    audio_array = np.array(tts.tts(text, speaker=current_speaker, language=detected_language, split_sentences=True))
    
    # Convert to int16 and scale
    audio_array = (audio_array * 32767).astype(np.int16)
    print(f"Full audio shape: {audio_array.shape}, dtype: {audio_array.dtype}")

    # Save full audio
    sf.write("full_audio.wav", audio_array, sr)
    print("Saved full audio to full_audio.wav")

    chunk_size = int(sr * 0.5)  # 0.5 second chunks
    print(f"Chunk size: {chunk_size}")

    full_output = np.array([], dtype=np.int16)

    for i in range(0, len(audio_array), chunk_size):
        state = get_state()
        if state["speak_mode"] == "clear":
            print("Speak mode cleared. Stopping.")
            yield (sr, np.array([], dtype=np.int16)), state
            return

        chunk = audio_array[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}, shape: {chunk.shape}")

        if len(chunk) > 0:
            # Update audio state
            current_audio = get_audio_state().get("audio_array", np.array([], dtype=np.int16))
            updated_audio = np.concatenate([current_audio, chunk])
            update_audio_state({"audio_array": updated_audio})
            
            # Update state
            state["speak_mode"] = "speaking"
            state["spoken"] = text[:int(len(text) * ((i + chunk_size) / len(audio_array)))]
            update_state(state)
            
            print(f"Updated state: {state}")
            
            # Save intermediate chunk
            sf.write("intermediate_chunk.wav", chunk, sr)
            print(f"Saved intermediate chunk {i//chunk_size + 1} to intermediate_chunk.wav")

            full_output = np.concatenate([full_output, chunk])
            yield (sr, chunk), state

    state = get_state()
    state["speak_mode"] = "finished"
    state["spoken"] = text
    update_state(state)
    print(f"Final state: {state}")
    print("Done speaking.")

    # Save entire yielded audio
    sf.write("full_output.wav", full_output, sr)
    print("Saved entire yielded audio to full_output.wav")

    yield (sr, np.array([], dtype=np.int16)), state


# def speak(text: str, state: Dict, speaker: str, language: str, second_speaker: str=SPEAKERS[0], second_language: str="ru") -> Iterator[Tuple[Tuple[int, np.ndarray], Dict]]:
#     """Generate and stream TTS audio using Coqui TTS with diarization."""
#     state = get_state()
#     print(f"SPEAK: state: {state}")
#     text = state["response"]
#     mode = state["speak_mode"]
#     sr = tts.synthesizer.output_sample_rate
#     chunk_duration = 0.25
#     chunk_size = int(sr * chunk_duration)
#     if text and len(text.split()) < 3 and not (text.endswith((".", "?", "!"))) or mode in ("clear"):
#         state["speak_mode"] = "wait"
#         return None

#     else:
#         sentences = [sentence.strip() for sentence in re.split(r"([.?!])", text) if sentence.strip()]
#         sentences = ["".join(sentences[i : i + 2]) for i in range(0, len(sentences), 2)]
#         sentences = [*sentences, ""]
#         spoken = state.get("spoken", "")
#         for sentence in sentences:
#             state = get_state()
#             if (
#                 sentence
#                 and not (spoken.strip() and sentence.startswith(spoken.strip()))
#                 and state["speak_mode"] != "clear"
#             ):
#                 language = detect(text)
#                 speaker = second_speaker if detect(text) == second_language else speaker
#                 audio_buffer = (
#                     np.array(tts.tts(sentence, speaker=speaker, language=language, split_sentences=False)) * 32767
#                 ).astype(np.int16)
#                 spoken += sentence
#                 update_state({"spoken": spoken})
#                 sf.write("output.wav", audio_buffer, sr)
#                 console.print(f"Speaking: {sentence}", style="bold white on  blue")
#                 state = get_state()
#                 console.print(f"STATE: {state}", style="bold white on blue")
#                 # concatenate and yield audio chunks at 0.25s intervals
#                 audio_out = get_audio().setdefault("audio_array", np.array([0], dtype=np.int16))
#                 audio_out = np.concatenate([audio_out, audio_buffer])
#                 update_audio({"audio_array": audio_out})
#                 for i in range(0, len(audio_buffer), chunk_size):
#                     if state["speak_mode"] == "clear":
#                         return None
#                     new_audio = audio_buffer[i : i + chunk_size]
#                     yield (sr, new_audio), state
#                 update_audio({"audio_array": audio_out})
              
#                 # yield (sr, audio_buffer), state
#                 # audio_array = np.concatenate([audio_array, new_audio.astype(np.int16)])
#                 # update_state({"speak_mode": "speaking", "spoken": spoken})
#                 # while len(audio_buffer) >= chunk_size:
#                 #     chunk = audio_buffer[:chunk_size]
#                 #     audio_buffer = audio_buffer[chunk_size:]
#                 #     if get_state()["speak_mode"] == "clear":
#                 #         return
#                 #     yield (sr, chunk), state

#             elif state["speak_mode"] == "clear":
#                 return  None
#             else:
#                 continue
#         state = get_state()
#         if not state.get("uncommitted", "") or state.get("spoken").endswith(state.get("uncommitted")):
#             state["speak_mode"] = "finished"
#             update_state({"speak_mode": "finished", "uncommitted": "", "spoken": ""})
#             console.print("Done speaking.", style="bold white on blue")
#         # else:
#         #     state["speak_mode"] = "speaking"
#         #     update_state({"speak_mode": "speaking"})
