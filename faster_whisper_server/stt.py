from collections.abc import Iterator
from copy import copy
from functools import partial
from multiprocessing.managers import BaseManager, BaseProxy
import os
import re
import threading
from time import sleep, time
from typing import Any, ClassVar, Dict, Literal, Tuple, TypedDict  # noqa: UP035

import gradio as gr
from gradio import Timer
import httpx
from lager import log
from langdetect import detect
from mbodied.agents.config import AgentConfig, CompletionConfig, Guidance
import numpy as np
from openai import OpenAI

# from pyannote.audio import Pipeline
from rich.console import Console
import soundfile as sf
from TTS.api import TTS

from faster_whisper_server.audio_config import AudioConfig
from faster_whisper_server.audio_task import (
    TaskConfig,
    get_audio_state,
    handle_audio_stream,
    update_audio_state,
)
from faster_whisper_server.speech import (
    SPEAKERS,
    get_speech_settings,
    get_speech_state,
    speak,
    update_speech_settings,
    update_speech_state,
)
from faster_whisper_server.transition import get_state, process_state, update_state
from faster_whisper_server.utils.colors import mbodi_color

WEBSOCKET_URI = "http://localhost:7543/v1/audio/transcriptions"

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


SYSTEM_PROMPT = """
You are a brand new assistant
and we are demonstrating your instruction following capabilities. Note that you are in a crowded room and there may be background noise.
"""

# def find_min_overlap(a: str, b: str) -> str:
#     """Find the minimum overlap where the end of 'a' matches the start of 'b'."""
#     a, b = a.lower(), b.lower()
#     for i in range(1, min(len(a), len(b)) + 1):
#         if a.endswith(b[:i]):
#             return b[:i]
#     return ""



os.environ["CUDA_VISIBLE_DEVICES"] = "6"
if gr.NO_RELOAD:
    task = TaskConfig()
    prelude_config = AgentConfig(
        auth_token="mbodi-demo-1",
        base_url="https://api.mbodi.ai/v1",
        completion=CompletionConfig(
            post_process=lambda q,r: q if r == "Yes" else "",
            prompt= lambda q: f"{DETERMINE_INSTRUCTION_PROMPT}{q}",
        )
    )

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
        api_key=prelude_config.auth_token, 
        # model_kwargs={"base_url": task.agent_base_url},
        context=SYSTEM_PROMPT
    )


NOT_A_COMPLETE_INSTRUCTION = "Not a complete instruction..."


console = Console()
print = console.print  # noqa: A001
gprint = partial(console.print, style="bold green on white")


def aprint(*args, **kwargs) -> None:
    console.print("ACT: ", *args, style="bold blue", **kwargs)


yprint = partial(console.print, style="bold yellow")
WAITING_FOR_NEXT_INSTRUCTION = "Waiting for next instruction..."

PredInstrMode = Literal["predict", "repeat", "clear", "wait"]
ActMode = Literal["acting", "repeat", "clear", "wait"]
SpeakMode = Literal["speaking", "wait", "clear"]



def act(instruction: str, last_response: str, last_tps: str, state: Dict | str) -> Iterator[Tuple[str, Dict]]:  # noqa: UP006
    state = get_state()
    instruction = state["instruction"]
    mode = state["act_mode"]
    if mode in ("clear", "wait"):
        update_state({"act_mode": "wait"})
        return "", last_tps, get_state()

    if mode == "repeat":
        update_state({"act_mode": "repeat"})
        return last_response, last_tps, state

    if len(agent.history()) > 10:
        agent.forget_after(2)

    tic = time()
    total_tokens = 0
    response = ""
    instruction = instruction + "\n Answer briefly and concisely."
    log.debug(f"Following instruction: {instruction}")
    last_response = ""
    for text in agent.act_and_stream(instruction=instruction, model="astroworld"):
        total_tokens = len(response.split())
        elapsed_time = time() - tic
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        response += text
        aprint(f"Response: {response}")
        if response and response == last_response:
            update_state({"act_mode": "acting"})
            return response, f"TPS: {tokens_per_sec:.4f}", get_state()
        update_state({"act_mode": "acting"})
        yield response, f"TPS: {tokens_per_sec:.4f}", get_state()
    update_state({"act_mode": "acting"})
    return response, f"TPS: {tokens_per_sec:.4f}", get_state()


console = Console()


def find_overlap(endswith: str, startswith: str) -> str:
    """Find the overlap where the end of 'endswith' matches the start of 'startswith'."""
    endswith = endswith or ""
    startswith = startswith or ""
    startswith, endswith = startswith.lower(), endswith.lower()
    if not endswith or not startswith:
        yprint(f"Endswith: {endswith}, Startswith: {startswith}, Returning empty string.")
        return ""
    for i in range(min(len(endswith), len(startswith)), 0, -1):
        if endswith.endswith(startswith[:i]):
            yprint(f"E: {endswith}, S: {startswith}, Overlap: {endswith[:i]}")
            return startswith[:i]
    return ""




def create_gradio_demo(config: AudioConfig, task_config: TaskConfig) -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue=mbodi_color,
            secondary_hue="stone",
        ),
        title="Personal Assistant",
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
        gr.Markdown(value="## Personal Assistant", render=True)
        with gr.Row():
            audio = gr.Audio(
                label="Audio Input",
                type="numpy",
                sources=["microphone"],
                streaming=True,
                interactive=True,
            )
            audio_out = gr.Audio(streaming=True, autoplay=True, label="Output", visible=True, render=True)
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
                second_speaker_language =gr.Dropdown(
                    choices=["en", "es", "fr", "de", "it", "ja", "ko", "nl", "pl", "pt", "ru", "zh"],
                    label="Second Speaker Language",
                    value=task.SECOND_LANGUAGE,
                    interactive=True,
                )
            with gr.Column():
                transcription = gr.Textbox(label="Transcription_raw", interactive=False, visible=False)
                transcription_persistant = gr.Textbox(label="Transcription", interactive=True, visible=True)
                transcription_tps = gr.Textbox(label="TPS Raw", placeholder="No data yet...", visible=False)
                transcription_tps_persistant = gr.Textbox(label="TPS", placeholder="No data yet...", visible=True)
                instruction = gr.Text(label="Instruction_raw", visible=False, value="", render=True, interactive=False)
                instruction_persistant = gr.Text(
                    label="Instruction", visible=True, value="", render=True, interactive=True
                )
                clear_button = gr.Button(value="Clear", render=True, variant="stop")
            with gr.Column():
                response_raw = gr.Text(label="agent_response", value="", visible=False)
                response = gr.Text(label="uncommitted", value="", visible=False)
                response_persistant = gr.Text(label="Response", value="", visible=True)
                response_tps = gr.Text(label="TPS response raw", placeholder="No data yet...", visible=False)
                response_tps_persistant = gr.Text(
                    label="Response TPS", placeholder="No data yet...", visible=True, interactive=False
                )

            def persistant_transcription(transcription: str, persistant: str) -> str:
                state = get_state()
                if any(state[k] == "clear" for k in ["pred_instr_mode", "act_mode", "speak_mode"]):
                    return ""
                if (transcription and persistant in transcription) or (
                    transcription and persistant == "Not enough audio yet."
                ):
                    return transcription
                return persistant

            def persist_transcription_tps(transcription_tps: str, persistant: str) -> str:
                state = get_state()
                if any(state[k] == "clear" for k in ["pred_instr_mode", "act_mode", "speak_mode"]):
                    return ""
                if transcription_tps and persistant in transcription_tps:
                    return transcription_tps
                return persistant

            def persist_response_tps(response_tps: str, persistant: str) -> str:
                state = get_state()
                if any(state[k] == "clear" for k in ["pred_instr_mode", "act_mode", "speak_mode"]):
                    return ""

                if response_tps and persistant in response_tps:
                    return response_tps
                return persistant

            def persist_instruction(instruction: str, persistant: str) -> str:
                state = get_state()
                if any(state[k] == "clear" for k in ["pred_instr_mode", "act_mode", "speak_mode"]):
                    return ""

                if (instruction and persistant in instruction) or (
                    instruction and persistant == NOT_A_COMPLETE_INSTRUCTION
                ):
                    return instruction
                return persistant

            def persist_response(response: str, persistant: str) -> str:
                state = get_state()
                # response = state.get("uncommitted", "")
                if any(state[k] == "clear" for k in ["pred_instr_mode", "act_mode", "speak_mode"]):
                    return ""
                if response and persistant in response:
                    return response
                return persistant

            transcription_tps.change(
                persist_transcription_tps,
                inputs=[transcription_tps, transcription_tps_persistant],
                outputs=[transcription_tps_persistant],
                every=0.5,
            )
            response_tps.change(
                persist_response_tps,
                inputs=[response_tps, response_tps_persistant],
                outputs=[response_tps_persistant],
                every=0.5,
            )
            instruction.change(
                persist_instruction,
                inputs=[instruction, instruction_persistant],
                outputs=[instruction_persistant],
                every=0.5,
            )
            response.change(
                persist_response,
                inputs=[response, response_persistant],
                outputs=[response_persistant],
                every=0.5,
            )
            transcription.change(
                persistant_transcription,
                inputs=[transcription, transcription_persistant],
                outputs=[transcription_persistant],
                every=0.5,
            )
            audio_state = gr.State(
                {
                    "stream": np.array([]),
                    "model": model_dropdown.value,
                    "temperature": 0.0,
                    "endpoint": task_config.TRANSCRIPTION_ENDPOINT,
                }
            )
            agent_state = gr.State(
                {
                    "pred_instr_mode": "predict",
                    "act_mode": "wait",
                    "speak_mode": "wait",
                    "transcription": "",
                    "instruction": "",
                    "uncommitted": "",
                }
            )

            def stream_audio(
                audio: tuple[int, np.ndarray],
                audio_state: dict,
                model: str,
            ) -> Iterator[tuple[dict, str, str]]:
                audio_state = get_audio_state()
                audio_state["model"] = model
                audio_settings = get_speech_settings()
                for state, transcription, transcription_tps in handle_audio_stream(
                    audio,
                    audio_state,
                    0.0,
                    http_client,
                    audio_settings,
                ):
                    update_audio_state(state)
                    yield state, transcription, transcription_tps

            audio.stream(
                fn=stream_audio,
                inputs=[
                    audio,
                    audio_state,
                    model_dropdown,
                ],
                outputs=[audio_state, transcription, transcription_tps],
            ).then(
                process_state,
                inputs=[transcription, instruction, response_raw, agent_state, first_speaker_language, second_speaker_language],
                outputs=[agent_state, transcription, instruction, response],
            )
            transcription.change(
                weak_agent.act,
                inputs=[transcription, instruction, agent_state, first_speaker_language, second_speaker_language],
                outputs=[response, agent_state],
            ).then(
                process_state,
                inputs=[transcription, instruction, response_raw, agent_state],
                outputs=[agent_state, transcription, instruction, response],
            )
            # is_speaking = gr.Checkbox(label="Speak", value=False, render=True, visible=False)
            # instruction.change(
            #     act,
            #     inputs=[instruction, response_raw, response_tps, agent_state],
            #     outputs=[response_raw, response_tps, agent_state],
            # )

            # response.change(
            #     speak,
            #     inputs=[response, agent_state, first_speaker_name, first_speaker_language],
            #     outputs=[audio_out, agent_state],
            #     trigger_mode="always_last",
            # ).then(

            gr.on(
                response.change,
                speak,
                inputs=[response, agent_state, first_speaker_name, first_speaker_language],
                outputs=[audio_out, agent_state],
                trigger_mode="always_last",
                every=0.5,
            )

            # then(
            #     speak,
            #     inputs=[response, agent_state, first_speaker_name, first_speaker_language],
            #     outputs=[audio_out, agent_state],
            #     trigger_mode="always_last",
            # )
        def clear(_) -> None:  # noqa
            update_state(
                {

                    "pred_instr_mode": "clear",
                    "act_mode": "clear",
                    "speak_mode": "clear",
                    "transcription": "",
                    "instruction": "",
                    "response": "",
                    "spoken": "",
                    "audio_array": np.array([0], dtype=np.int16),
                    "uncommitted": "",
                }
            )
            update_audio_state({"stream": np.array([])})
            return (
                gr.Audio(streaming=True, autoplay=True, label="Output", visible=True, render=True, value=(
                    22050,
                    np.array([0], dtype=np.int16))),
                get_state(),
                get_audio_state(),
                "", "", "",
                "", "", "",
                "", "", "",
            )
        clear_button.click(clear, inputs=[clear_button], outputs=[
            audio_out, agent_state, audio_state, 
            transcription, instruction, response, 
            transcription_persistant, instruction_persistant, response_persistant,
            transcription_tps, transcription_tps_persistant, response_tps_persistant
        ])

        # audio_out.stream(speak, inputs=[response, agent_state, first_speaker_name, first_speaker_language], outputs=[audio_out, agent_state], trigger_mode="always_last")
        demo.load(update_model_dropdown, inputs=None, outputs=[model_dropdown])
        return demo


demo = create_gradio_demo(AudioConfig(), TaskConfig())

if __name__ == "__main__":
    # debug = sys.argv[-1] if len(sys.argv) > 1 else "INFO"
    from rich.logging import RichHandler

    log.add(RichHandler(), level="DEBUG")
    demo.queue().launch(
        server_name="0.0.0.0", share=False, show_error=True, debug=True, root_path="/instruct", server_port=7862
    )
