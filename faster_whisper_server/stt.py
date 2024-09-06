import collections
from collections.abc import Iterator
from functools import partial
import os
import re
import threading
from time import sleep, time
from typing import Any, Literal

import gradio as gr
from gradio import Timer
import httpx
from lager import log

# from langdetect import detect
import numpy as np
from openai import OpenAI

# from pyannote.audio import Pipeline
from rich.console import Console
import soundfile as sf
from TTS.api import TTS

from faster_whisper_server.audio_config import AudioConfig
from faster_whisper_server.audio_task import TaskConfig, handle_audio_stream
from faster_whisper_server.colors import mbodi_color

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
SYSTEM_PROMPT = """
You are a brand new assistant
and we are demonstrating your instruction following capabilities. Note that you are in a crowded room and there may be background noise.
"""


### Weak Agent ###
DETERMINE_INSTRUCTION_PROMPT = """
Determine if the statement after POTENTIAL INSTRUCTION is an instruction for a robot to take action, other statement, or incomplete statement. Answer "Yes", "No", or "Incomplete".

PORENTIAL INSTRUCTION:
"""


_agent_state: dict[str, Any] = {
    "pred_instr_mode": "predict",
    "act_mode": "wait",
    "speak_mode": "wait",
    "transcription": "",
    "instruction": "",
    "response": "",
    "spoken": "",
    "moving": False,
    "audio_array": np.array([0], dtype=np.int16),
    "audio_finish_time": float("inf"),
    "speech_fragments": 0,
}

_audio_state: dict[str, Any] = {
    "stream": np.array([]),
    "model": "Systran/faster-distil-whisper-large-v3",
    "temperature": 0.0,
    "endpoint": "/audio/transcriptions"
}

# Lock for thread-safe access to global state
state_lock = threading.Lock()
audio_lock = threading.Lock()
def get_state() -> dict[str, Any]:
    with state_lock:
        return _agent_state

def update_state(updates: dict[str, Any]) -> None:
    with state_lock:
        _agent_state.update(updates)

def clear_states() -> None:
    with state_lock:
        _agent_state.clear()
        _agent_state.update({
            "pred_instr_mode": "predict",
            "act_mode": "wait",
            "speak_mode": "wait",
            "transcription": "",
            "instruction": "",
            "response": "",
            "first_act": False,
        })
    with audio_lock:
        _audio_state.update({
            "stream": np.array([]),
            "model": "Systran/faster-distil-whisper-large-v3",
            "temperature": 0.0,
            "endpoint": "/audio/transcriptions"
        })
def get_audio() -> dict[str, Any]:
    with audio_lock:
        return _audio_state

def update_audio(updates: dict[str, Any]) -> None:
    with audio_lock:
        _audio_state.update(updates)


os.environ["CUDA_VISIBLE_DEVICES"] = "6"
if gr.NO_RELOAD:
    task = TaskConfig()
    tts = TTS(task.TTS_MODEL, gpu=True)

    # Initialize models

    TIMEOUT = httpx.Timeout(timeout=task.TIMEOUT_SECONDS)
    base_url = "https://api.mbodi.ai/audio/v1"
    http_client = httpx.Client(base_url=base_url, timeout=TIMEOUT)
    openai_client = OpenAI(base_url=f"{base_url}", api_key="cant-be-empty")
    http_client = httpx.Client(base_url=task.TRANSCRIPTION_ENDPOINT, timeout=TIMEOUT)
    from mbodied.agents import LanguageAgent

    agent = LanguageAgent(
        api_key=task.agent_token, model_kwargs={"base_url": task.agent_base_url}, context=SYSTEM_PROMPT
    )
    weak_agent = LanguageAgent(
        api_key=task.agent_token, model_kwargs={"base_url": task.agent_base_url}, context=SYSTEM_PROMPT
    )


NOT_A_COMPLETE_INSTRUCTION = "Not a complete instruction..."


console = Console()
print = console.print  # noqa: A001
gprint = partial(console.print, style="bold green on white")


def aprint(*args, **kwargs):
    console.print("ACT: ", *args, style="bold blue", **kwargs)


yprint = partial(console.print, style="bold yellow")
WAITING_FOR_NEXT_INSTRUCTION = "Waiting for next instruction..."

PredInstrMode = Literal["predict", "repeat", "clear", "wait"]
ActMode = Literal["acting", "repeat", "clear", "wait"]
SpeakMode = Literal["speaking", "wait", "clear"]


def predict_instruction(transcription: str, last_instruction: str) -> Iterator[tuple[str, dict]]:  # noqa: UP006
    update_state({"transcription": transcription})
    state = get_state()

    state = get_state()
    print(f"PREDICTION: state: {state}")
    mode = state["pred_instr_mode"]
    if mode == "clear" or not transcription or not transcription.strip():
        clear_states()
        return ""
    if mode == "repeat":
        update_state({"pred_instr_mode": "repeat", "instruction": last_instruction})
        return last_instruction
    if not state.get("transcription") or (len(state["transcription"].split()) < 2 and not state["transcription"].endswith((".", "?", "!"))):
        return NOT_A_COMPLETE_INSTRUCTION

    yprint(f"Text: {transcription}, Mode Predict: {state}, Last instruction: {last_instruction}")

    weak_agent.forget(everything=True)
    full_instruction = DETERMINE_INSTRUCTION_PROMPT + "\n" + transcription
    response =  weak_agent.act(instruction=full_instruction, model="astroworld", extra_body={"guided_choice": ["Yes", "No", "Incomplete Statement"]})
    if (
        response in ("Yes", "No")
    ):
        gprint(f"{response} {transcription}")
        update_state({"pred_instr_mode": "repeat"})
        update_state({"instruction": transcription})
        update_state({"act_mode": "acting"})
        gprint(f"transcription: {transcription}")
        if response == "Yes":
            update_state({"moving": True})
        else:
            update_state({"moving": False})
        return transcription
    else:
        print(f"Instruction: {transcription} is not a complete instruction.")
        update_state({"pred_instr_mode": "predict"})
        yield NOT_A_COMPLETE_INSTRUCTION

        yprint(weak_agent.act(instruction="Why wasn't it a complete instruction?", model="astroworld"))
        update_state({"pred_instr_mode": "predict"})
        return NOT_A_COMPLETE_INSTRUCTION

speak_dequeue = collections.deque(maxlen=100)

def act(instruction: str, last_response: str, last_tps: str) -> Iterator[tuple[str, dict]]:  # noqa: UP006
    state = get_state()
    print(f"ACT: state: {state}")
    instruction = state["instruction"]
    mode = state["act_mode"]
    if mode in ("clear", "wait"):
        return "", last_tps

    if mode == "repeat":
        update_state({"act_mode": "repeat"})
        return last_response, last_tps

    if len(agent.history()) > 10:
        agent.forget_after(2)

    tic = time()
    total_tokens = 0
    response = ""
    instruction = instruction + "\n Answer briefly and concisely."
    aprint(f"Following instruction: {instruction}")
    last_response = ""
    for text in agent.act_and_stream(instruction=instruction, model="astroworld"):
        total_tokens = len(response.split())
        elapsed_time = time() - tic
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        response += text
        aprint(f"Response: {response}")
        if response and response == last_response:
            update_state({"act_mode": "repeat", "response": response, "speak_mode": "speaking"})    
            return response, f"TPS: {tokens_per_sec:.4f}"
        update_state({"act_mode": "acting","response": response, "speak_mode": "speaking"})
        yield response, f"TPS: {tokens_per_sec:.4f}"
    update_state({"act_mode": "repeat", "response": response, "speak_mode": "speaking"})
    return response, f"TPS: {tokens_per_sec:.4f}"

def speak(text: str, speaker: str, language: str) -> Iterator[tuple[bytes, str, dict]]:
    """Generate and stream TTS audio using Coqui TTS with diarization."""
    state = get_state()
    console.print(f"speak STATE: {state}")
    text = state["response"]
    mode = state["speak_mode"]
    sr = tts.synthesizer.output_sample_rate
    spoken = state.get("spoken", "")

    console.print(f"Text: {text}, Mode Speak: {mode} spoken {spoken}", style="red")
    if not text:
        return


        # Check if this is a new response and reset spoken_idx
    # if state["spoken"] == "" and state.setdefault("spoken_idx", 0) > 0:
    #     console.print("New response detected, resetting spoken_idx to 0.", style="bold green")
    #     update_state({"spoken_idx": 0})
    # Handle incomplete sentences more gracefully
    if text and len(text.split()) < 2 and not (text.endswith((".", "?", "!"))) and state["act_mode"] != "repeat":
        return

    if mode == "clear":
        clear_states()
        return
    if mode == "wait":
        return


        # Split the text and keep the punctuation
    sentences = [sentence.strip() + punct for sentence, punct in re.findall(r"([^.!?]*)([.!?])", text) if sentence.strip()]
        # sentences = [*sentences]
        # update_state({"sentences": sentences})
    # spoken_idx = state.get("spoken_idx", 0)

    console.print(f"Sentences: {sentences}, spoken_idx:, spoken: {spoken}", style="red")

    # Speak each sentence
    for idx, sentence in enumerate(sentences):
        if sentence and sentence not in spoken:
            print(f"Speaking sentence: {sentence}")

            # Synthesize speech for the sentence
            audio_array = (np.array(tts.tts(sentence, speaker=speaker, language=language, split_sentences=False)) * 32767).astype(np.int16)

            if state["speak_mode"] == "clear":
                clear_states()
                return

            # Log the sentence and mark the state as speaking
            console.print(f"SPEAK Text: {sentence}, Mode Speak: {mode}", style="bold white on blue")
            spoken += sentence
            audio_seconds = len(audio_array) / float(sr)
            console.print(f"Audio seconds: {audio_seconds}", style="bold white on blue")
            # Update state
            update_state({
                "speak_mode": "speaking",
                "spoken": spoken,
                "audio_array": audio_array,
                "act_mode": "repeat",
                "spoken_idx": idx + 1,  # Move to the next sentence
                "audio_finish_time": time() + audio_seconds,
            })
            # Save audio to file and yield audio output
            f = f"out{idx}.wav"
            sf.write(file=f, data=audio_array, samplerate=sr)
            return f

    # if spoken_idx >= len(sentences) - 1:

    update_state({"speak_mode": "wait", "spoken": "", "audio_array": np.array([0], dtype=np.int16), "uncommitted": "", "response": "",
                    "act_mode": "wait", 
                    "audio_finish_time": float("inf"),
                    "first_act": False,
                    "pred_instr_mode": "predict", "transcription": "", "instruction": "",
                    "sentences": [], "spoken_idx": 0,     "stream": np.array([])})

    # console.print("Done speaking.", style="bold white on blue")
    # Final check for completion
    # if spoken_idx >= len(sentences) - 1:
    #     update_state({"speak_mode": "wait", "spoken": "", "audio_array": np.array([0], dtype=np.int16),
    #      "uncommitted": "", "act_mode": "wait",
    #     "sentences": [], "spoken_idx": 0, "audio_finish_time": 0,     "stream": np.array([])})
    #     console.print("Done speaking.", style="bold white on blue")
    # if get_state()["speak_mode"] == "clear":
    #     update_state({"speak_mode": "wait", "spoken": "", "audio_array": np.array([0], dtype=np.int16),
    #      "uncommitted": "", "response": "", "act_mode": "wait", "pred_instr_mode": "predict",
    #       "transcription": "", "instruction": "", "sentences": [], "spoken_idx": 0, "audio_finish_time": 0, "stream": np.array([])})




def create_gradio_demo(config: AudioConfig, task_config: TaskConfig) -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue=mbodi_color,
            secondary_hue="stone",
        ),
        title="Personal Assistant",
        delete_cache=[0,0]
    ) as demo:

        def update_model_dropdown() -> gr.Dropdown:
            models = openai_client.models.list().data
            model_names: list[str] = [model.id for model in models]
            recommended_models = {model for model in model_names if model.startswith("Systran")}
            other_models = [model for model in model_names if model not in recommended_models]
            model_names = list(recommended_models) + other_models
            return gr.Dropdown(
                choices=model_names,
                label="Model",
                value=config.whisper.model,
            )

        clear_button = gr.Button(value="Clear", render=True, variant="primary")

        with gr.Row():
                audio = gr.Audio(
                    label="Audio Input",
                    type="numpy",
                    sources=["microphone"],
                    streaming=True,
                    interactive=True,
                )
                audio_out = gr.Audio(streaming=False, autoplay=True, label="Output", visible=True, render=True, type="filepath")
                model_dropdown = gr.Dropdown(
                    choices=[config.whisper.model],
                    label="Model",
                    value=config.whisper.model,
                    interactive=True,
                )
        with gr.Row():
            with gr.Column():
                first_speaker_name = gr.Dropdown(
                    label="First Speaker Name",
                    choices=SPEAKERS,
                    value=task.FIRST_SPEAKER,
                    interactive=True,
                    )
                first_speaker_language = gr.Dropdown(
                    choices=["en", "es", "fr", "de", "it", "ja", "ko", "nl", "pl", "pt", "ru", "zh"],
                    label="First Speaker Language",
                    value=task.FIRST_LANGUAGE,
                    interactive=True,
                )
                second_speaker_name = gr.Dropdown(
                    label="Second Speaker Name",
                    choices=SPEAKERS,
                    value=task.SECOND_SPEAKER,
                    interactive=True,
                )
                second_speaker_language = gr.Dropdown(
                    choices=["en", "es", "fr", "de", "it", "ja", "ko", "nl", "pl", "pt", "ru", "zh"],
                    label="Second Speaker Language",
                    value=task.SECOND_LANGUAGE,
                    interactive=True,
                )
            with gr.Column():
                transcription = gr.Textbox(label="Transcription", interactive=False)
                transcription_tps = gr.Textbox(label="TPS", placeholder="No data yet...", visible=True)
                instruction = gr.Text(label="Instruction", visible=True, value="", render=True, interactive=True)
            with gr.Column():
                response = gr.Text(label="Response", value="", visible=True)
                response_tps = gr.Text(label="TPS", placeholder="No data yet...", visible=True)
            audio_state = gr.State(
                {
                    "stream": np.array([]),
                    "model": model_dropdown.value,
                    "temperature": 0.0,
                    "endpoint": task_config.TRANSCRIPTION_ENDPOINT,
                }
            )
            def stream_audio(
                audio: tuple[int, np.ndarray],
                audio_state: dict,
                model: str,
            ) -> Iterator[tuple[str, str]]:
                # print(f"THIS IS THE AUDIO STATE: {audio_state}")
                audio_state = get_audio()
                audio_state["model"] = model

                updated_stream = False
                stream = get_state().get("stream")
                if stream is not None:
                    audio_state["stream"] = stream
                    updated_stream = True

                for state, transcription, transcription_tps in handle_audio_stream(
                    audio,
                    audio_state,
                    0.0,
                    http_client,
                ):
                    update_audio(state)
                    if "Not enough audio yet." in transcription:
                        transcription = ""
                    yield get_audio(), transcription, transcription_tps

                    if updated_stream:
                        update_state({"stream": None})

            audio.stream(
                fn=stream_audio,
                inputs=[
                    audio,
                    audio_state,
                    model_dropdown,
                ],
                outputs=[audio_state, transcription, transcription_tps]
            )
            transcription.change(
                predict_instruction,
                inputs=[transcription, instruction],
                outputs=[instruction],
            )
            instruction.change(
                act,
                inputs=[instruction, response, response_tps],
                outputs=[response, response_tps],
            )
            response.change(
                speak,
                inputs=[response, first_speaker_name, first_speaker_language],
                outputs=[audio_out],
                # trigger_mode="always_last",
            )

            # def increment_speech():
            #     state = get_state()
            #     num_speech_fragments = state.get("speech_fragments", 0)
            #     if state.get("act_mode") == "repeat" and not state.get("first_act"):
            #         update_state({"first_act": True})
            #         num_speech_fragments += 1
            #         return num_speech_fragments
            #     if time() < state.get("audio_finish_time", float("inf")):
            #         return num_speech_fragments

            #     num_speech_fragments += 1
            #     yprint(f"Incrementing speech fragments to {num_speech_fragments}")
            #     update_state({"speech_fragments": num_speech_fragments, "audio_finish_time": float("inf")})
            #     return num_speech_fragments

            # speech_fragments = gr.Number(label="Speech Fragments", value=increment_speech, visible=False, every=1)
            # gr.on(
            #     [speech_fragments.change],
            #     speak,
            #     inputs=[response, second_speaker_name, second_speaker_language],
            #     outputs=[audio_out],
            # )
            speak_button = gr.Button(value="Speak", render=True, variant="secondary")
            gr.on(
                [speak_button.click],
                speak,
                inputs=[response, second_speaker_name, second_speaker_language],
                outputs=[audio_out],
            )
            def clear_button_click():
                clear_states()
                return "", "", ""

            clear_button.click(
                clear_button_click,
                inputs=[],
                outputs=[transcription, instruction, response],
            )
        demo.load(update_model_dropdown, inputs=None, outputs=[model_dropdown])
        return demo


demo = create_gradio_demo(AudioConfig(), TaskConfig())

if __name__ == "__main__":
    from rich.logging import RichHandler

    log.add(RichHandler(), level="DEBUG")
    demo.queue().launch(
        server_name="0.0.0.0", share=False, show_error=True, debug=True, root_path="/instruct", server_port=7861
    )
