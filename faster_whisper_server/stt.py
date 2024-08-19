from collections.abc import Generator, Iterator
from functools import partial
import logging
import os
from pathlib import Path
import threading
from time import time
import traceback
from typing import Dict, Literal, Tuple

import gradio as gr
from gradio import Timer
import httpx
from lager import log

# from langdetect import detect
import numpy as np
from openai import OpenAI

# from pyannote.audio import Pipeline
from rich.console import Console
from rich.pretty import pprint
from TTS.api import TTS
from typing_extensions import TypedDict

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
Determine if the following is an instruction to follow or question to answer. It may be a partial instruction or random thought or anything else.
Answer  "Yes" if an only if it is a complete instruction to follow or question to answer. Otherwise, answer "No. 

Some examples of complete instructions:
"Tell me about the weather in New York."
"Tell me your name."
"Move the chair to the left."

NOTE: Any question or command that requires a response or action is considered an instruction.

If there are multiple instructions, answer "Yes" if any of them are complete instructions. Otherwise, answer "No."

POTENTIAL INSTRUCTION:
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
if gr.NO_RELOAD:
    task = TaskConfig()
    tts = TTS(task.TTS_MODEL, gpu=True)

    # Initialize models
    config = AudioConfig()
    TIMEOUT = httpx.Timeout(timeout=task.TIMEOUT_SECONDS)
    base_url = "https://api.mbodi.ai/audio/v1"
    http_client = httpx.Client(base_url=base_url, timeout=TIMEOUT)
    openai_client = OpenAI(base_url=f"{base_url}", api_key="cant-be-empty")
    http_client = httpx.Client(base_url=task.TRANSCRIPTION_ENDPOINT, timeout=TIMEOUT)
    from mbodied.agents.language import LanguageAgent

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


class State(TypedDict):
    pred_instr_mode: Literal["predict", "repeat", "clear", "wait"]
    act_mode: Literal["acting", "repeat", "clear", "wait"]
    speak_mode: Literal["speaking", "wait", "clear"]
    transcription: str
    instruction: str


class AudioState(TypedDict):
    stream: np.ndarray
    model: str
    temperature: float


def predict_instruction(transcription: str, last_instruction: str, state: Dict) -> Iterator[Tuple[str, Dict]]:  # noqa: UP006
    mode = state.pop("pred_instr_mode")
    if mode == "clear" or not transcription or not transcription.strip():
        return "", state.setdefault("pred_instr_mode", "wait")
    if mode == "repeat":
        return last_instruction, state

    if transcription == NOT_A_COMPLETE_INSTRUCTION:
        msg = "Instruction is not a complete instruction."
        raise ValueError(msg)

    yprint(f"Text: {transcription}, Mode Predict: {state}, Last instruction: {last_instruction}")

    weak_agent.forget(everything=True)
    full_instruction = DETERMINE_INSTRUCTION_PROMPT + "\n" + transcription
    if (
        weak_agent.act(instruction=full_instruction, model="astroworld", extra_body={"guided_choice": ["Yes", "No"]})
        == "Yes"
    ):
        gprint(f"Instruction: is a complete instruction. Returning {transcription}")
        return transcription, state.setdefault("pred_instr_mode", "repeat")
    else:
        print(f"Instruction: {transcription} is not a complete instruction.")
        yield NOT_A_COMPLETE_INSTRUCTION, state.setdefault("pred_instr_mode", "predict")

        yprint(weak_agent.act(instruction="Why wasn't it a complete instruction?", model="astroworld"))
        return NOT_A_COMPLETE_INSTRUCTION, state.setdefault("pred_instr_mode", "predict")


def act(instruction: str, last_response: str, last_tps: str, state: Dict | str) -> Iterator[Tuple[str, Dict]]:  # noqa: UP006
    aprint(f"Instruction: {instruction}, last response: {last_response}, state: {state}")
    mode = state.pop("act_mode")
    if mode in ("clear", "wait"):
        return "", last_tps, state.setdefault("act_mode", "wait")

    if mode == "repeat":
        return last_response, last_tps, state.setdefault("act_mode", "repeat")
    if mode not in ["acting", "wait"]:
        raise ValueError(f"Invalid mode: {mode}")

    if len(agent.history()) > 10:
        agent.forget_after(2)

    tic = time()
    total_tokens = 0
    response = ""
    instruction = instruction + "\n Answer briefly and concisely."
    gprint(f"Following instruction: {instruction}.")
    last_response = ""
    for text in agent.act_and_stream(instruction=instruction, model="astroworld"):
        total_tokens = len(response.split())
        elapsed_time = time() - tic
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        response += text
        aprint(f"Response: {response}")
        if response == last_response:
            return response, f"TPS: {tokens_per_sec:.4f}", "repeat"
        yield response, f"TPS: {tokens_per_sec:.4f}", "acting"


def speak(text: str, state: TaskConfig) -> Iterator[Tuple[bytes, str, Dict]]:  # noqa: UP006
    """Generate and stream TTS audio using Coqui TTS with diarization."""
    mode = state.pop("speak_mode")
    console.print(f"SPEAK Text: {text}, Mode Speak: {mode}")

    if text and len(text.split()) < 3:
        return b"", "", state.setdefault("speak_mode", "wait")

    if mode in ["clear", "wait"]:
        return b"", "", state.setdefault("speak_mode", "wait")

    sentences = text.split(". ")
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    sentences = [*sentences, ""]
    for sentence in sentences:
        if sentence:
            audio_array = (np.array(tts.tts(sentence, speaker=state.speaker, language=state.language)) * 32767).astype(
                np.int16
            )
            pprint(f"Speaking: {sentence}")
            yield audio_array.tobytes(), state.setdefault("speak_mode", "speaking")
        else:
            return "", "", state.setdefault("speak_mode", "wait")
    print("Done speaking.")


def process_state(instruction: str, response: str, state: State) -> State:
    print(f"Current state: {state}")

    # If any mode is "clear", reset all modes
    if any(state[mode] == "clear" for mode in ["pred_instr_mode", "act_mode", "speak_mode"]):
        return State(pred_instr_mode="predict", act_mode="wait", speak_mode="wait", transcription="", instruction="")

    # If no complete instruction yet
    if instruction == NOT_A_COMPLETE_INSTRUCTION:
        return State(
            pred_instr_mode="predict",
            act_mode="wait",
            speak_mode="wait",
            transcription=state["transcription"],
            instruction=state["instruction"],
        )

    # If we have an instruction but haven't acted on it
    if instruction and state["act_mode"] == "wait":
        return State(
            pred_instr_mode="repeat",
            act_mode="acting",
            speak_mode="wait",
            transcription=state["transcription"],
            instruction=instruction,
        )

    # If we're acting and have a response, but haven't started speaking
    if state["act_mode"] == "acting" and response and state["speak_mode"] == "wait":
        return State(
            pred_instr_mode="repeat",
            act_mode="repeat",
            speak_mode="speaking",
            transcription=state["transcription"],
            instruction=instruction,
        )

    # If we're speaking
    if state["speak_mode"] == "speaking":
        return State(
            pred_instr_mode="repeat",
            act_mode="repeat",
            speak_mode="speaking",
            transcription=state["transcription"],
            instruction=instruction,
        )

    # If we've finished speaking
    if state["speak_mode"] == "wait" and state["act_mode"] == "repeat" and state["pred_instr_mode"] == "repeat":
        return State(pred_instr_mode="predict", act_mode="wait", speak_mode="wait", transcription="", instruction="")

    # Default case: maintain current state
    return state


def create_gradio_demo(config: AudioConfig) -> gr.Blocks:
    audio = gr.Audio(
        label="Audio Input", type="numpy", sources=["microphone"], streaming=True, interactive=True, render=False
    )
    audio_out = gr.Audio(streaming=True, autoplay=True, label="Output", visible=False, render=False)
    temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.0, render=False)
    model_dropdown = gr.Dropdown(
        choices=[config.whisper.model],
        label="Model",
        value=config.whisper.model,
    )
    first_speaker_language = gr.Dropdown(
        choices=["en", "es", "fr", "de", "it", "ja", "ko", "nl", "pl", "pt", "ru", "zh"],
        label="First Speaker Language",
        value=task.FIRST_LANGUAGE,
    )
    second_speaker_language = gr.Dropdown(
        choices=["en", "es", "fr", "de", "it", "ja", "ko", "nl", "pl", "pt", "ru", "zh"],
        label="Second Speaker Language",
        value=task.SECOND_LANGUAGE,
    )
    first_speaker_name = gr.Dropdown(
        label="First Speaker Name",
        render=False,
        choices=SPEAKERS,
        value=task.FIRST_SPEAKER,
    )
    second_speaker_name = gr.Dropdown(
        label="Second Speaker Name",
        render=False,
        choices=SPEAKERS,
        value=task.SECOND_SPEAKER,
    )

    state = gr.State(
        State(pred_instr_mode="predict", act_mode="wait", speak_mode="wait", transcription="", instruction="")
    )
    audio_state = gr.State(
        AudioState(stream=np.array([]), model=config.whisper.model, task="transcribe", temperature=0.0)
    )
    task_config = gr.State(task, render=False)
    with gr.Blocks(
            theme=mbodi_color,
            title="Personal Assistant",
        ) as demo:
        @gr.on(
            [
                demo.load,
                model_dropdown.change,
                first_speaker_language.change,
                second_speaker_language.change,
                first_speaker_name.change,
                second_speaker_name.change,
            ],
            inputs=[
                model_dropdown,
                first_speaker_language,
                second_speaker_language,
                first_speaker_name,
                second_speaker_name,
                task_config,
                task_config,
            ],
            outputs=[
                model_dropdown,
                first_speaker_language,
                second_speaker_language,
                first_speaker_name,
                second_speaker_name,
                task_config,
                audio_state,
            ],
        )
        def update_dropdowns(
            model_dropdown: str,
            first_speaker_language: str,
            second_speaker_language: str,
            first_speaker_name: str,
            second_speaker_name: str,
            state: State,
            audio_state: AudioState,
        ) -> Tuple[str, str, str]:
            global tts
            if not tts:
                thread.join()
            models = openai_client.models.list().data
            model_names: list[str] = [model.id for model in models]
            recommended_models = {model for model in model_names if model.startswith("Systran")}
            other_models = [model for model in model_names if model not in recommended_models]
            model_names = list(recommended_models) + other_models
            model = gr.Dropdown(choices=model_names, label="Model", value=model_dropdown)
            first_speaker_language = gr.Dropdown(
                choices=["en", "es", "fr", "de", "it", "ja", "ko", "nl", "pl", "pt", "ru", "zh"],
                label="First Speaker Language",
                value=first_speaker_language,
                allow_custom_value=False,
            )
            second_speaker_language = gr.Dropdown(
                choices=["en", "es", "fr", "de", "it", "ja", "ko", "nl", "pl", "pt", "ru", "zh"],
                label="Second Speaker Language",
                value=second_speaker_language,
                allow_custom_value=False,
            )
            state.FIRST_LANGUAGE = first_speaker_language
            state.SECOND_LANGUAGE = second_speaker_language
            audio_state.model = model
            return (
                model,
                first_speaker_language,
                second_speaker_language,
                first_speaker_name,
                second_speaker_name,
                state,
                audio_state,
            )

        transcription = gr.Textbox(label="Transcription", placeholder=task.PLACE_HOLDER, render=False)
        transcription_tps = gr.Textbox(label="TPS", placeholder="No data yet...", render=False)

        instruction = gr.Text(label="Instruction", visible=False, value="", render=False)
        response = gr.Textbox(label="Response", placeholder="", visible=False, render=False)
        response_tps = gr.Textbox(label="TPS", placeholder="No data yet...", visible=False, render=False)
        gr.on(
            [audio.stream],
            partial(handle_audio_stream, http_client=http_client),
            inputs=[
                audio,
                audio_state,
                temperature_slider,
                audio_state,
            ],
            outputs=[audio_state, transcription, transcription_tps],
            trigger_mode="always_last",
        ).then(process_state, inputs=[instruction, response, state], outputs=[state], trigger_mode="always_last")

        timer = gr.Timer(value=0.5, render=True)
        gr.on([timer.tick], process_state, inputs=[state], outputs=[state], trigger_mode="always_last").then(
            act,
            inputs=[instruction, response, response_tps, state],
            outputs=[response, response_tps, state],
        ).then(process_state, inputs=[state], outputs=[state], trigger_mode="always_last")

        clear_button = gr.Button(value="Clear", render=False)

        should_instruct = gr.Checkbox(label="Instruct", value=False, visible=True, render=False)
        should_speak = gr.Checkbox(label="Speak", value=False, visible=True, render=False)

        @gr.on(
            [should_speak.change],
            inputs=[audio_out, should_speak],
            outputs=[audio_out],
        )
        def update_audio_out(is_checked: bool):
            return gr.Audio(streaming=True, autoplay=True, label="Output", visible=is_checked, render=is_checked)

        @gr.on(
            [should_instruct.change],
            inputs=[should_instruct],
            outputs=[instruction, response, response_tps],
        )
        def update_visibility(is_checked: bool):
            return (
                gr.Text(label="Instruction", visible=is_checked, render=is_checked),
                gr.Textbox(label="Response", placeholder="", visible=is_checked, render=is_checked),
                gr.Textbox(label="TPS", placeholder="", visible=is_checked, render=is_checked),
            )

        @gr.on(
            [clear_button.click],
            inputs=[clear_button],
            outputs=[state, audio_state, transcription, transcription_tps, instruction, response, response_tps],
        )
        def clear_everything(audio_state):
            return (
                State(pred_instr_mode="clear", act_mode="clear", speak_mode="clear", transcription="", instruction=""),
                AudioState(
                    stream=np.array([]),
                    model=audio_state["model"],
                    transcription="",
                    temperature=audio_state.get("temperature", 0.0),
                ),
                "",
                "",
                "",
                "",
                "",
            )

        with gr.Column():
            audio.render()
            temperature_slider.render()
        with gr.Column():
            model_dropdown.render()
            first_speaker_language.render()
            second_speaker_language.render()
            clear_button.render()

        with gr.Column():
            audio_out.render()
            transcription.render()
            should_speak.render()
            instruction.render()

        audio_out.stream(speak, inputs=[response, state], outputs=[audio_out, state], trigger_mode="always_last")

        return demo


demo = create_gradio_demo(AudioConfig())

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    import sys

    debug = sys.argv[-1] if len(sys.argv) > 1 else "INFO"
    from rich.logging import RichHandler

    log.add(RichHandler(), level=debug)
    demo.queue().launch(
        server_name="0.0.0.0", share=False, show_error=True, debug=True, root_path="/instruct", server_port=7861
    )
