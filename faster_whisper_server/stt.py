from collections.abc import Generator
from functools import partial
import logging
import os
from pathlib import Path
from time import time
import traceback
from typing import Iterator, Literal, Tuple

import gradio as gr
from gradio import Timer
from gradio.themes.utils import colors
import httpx
from httpx_sse import connect_sse
from langdetect import detect
from mbodied.agents import LanguageAgent
import numpy as np
from openai import OpenAI
from pyannote.audio import Pipeline
from rich.pretty import pprint
from TTS.api import TTS
from rich.console import Console
from faster_whisper_server.config import Config, Task
from faster_whisper_server.processing import audio_to_bytes

TRANSCRIPTION_ENDPOINT = "/audio/transcriptions"
TRANSLATION_ENDPOINT = "/audio/translations"
TIMEOUT_SECONDS = 180
TIMEOUT = httpx.Timeout(timeout=TIMEOUT_SECONDS)
WEBSOCKET_URI = "wss://api.mbodi.ai/audio/v1/transcriptions"
CHUNK = 1024
CHANNELS = 1
RATE = 16000
PLACE_HOLDER = "Loading can take 30 seconds if a new model is selected..."

logger = logging.getLogger(__name__)

from mbodied.types.language.control import HandControl
http_client = httpx.Client(base_url="https://api.mbodi.ai/audio/v1", timeout=TIMEOUT)
mbodi_color = colors.Color(
    c50="#fde8e8",
    c100="#f9b9b9",
    c200="#f58b8b",
    c300="#f15c5c",
    c400="#ee2e2e",
    c500="#ed3d5d",  # Main theme color
    c600="#b93048",
    c700="#852333",
    c800="#51171f",
    c900="#1e0a0a",
    c950="#0a0303",
    name="mbodi",
)

CUSTOM_CSS = """
#footer {display: none !important;}
"""
THEME = gr.themes.Soft(
    primary_hue=mbodi_color,
    secondary_hue="stone",
    neutral_hue="zinc",
    font_mono=["IBM Plex Mono", "ui-monospace", "Consolas", "monospace"],
)

device = "cuda:5"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# Initialize models
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")


def audio_task(file_path: str, endpoint: str, temperature: float, model: str) -> str:
    with Path(file_path).open("rb") as file:
        response = http_client.post(
            endpoint,
            files={"file": file},
            data={
                "model": model,
                "response_format": "text",
                "temperature": temperature,
            },
        )

    response.raise_for_status()
    yield response.text






SYSTEM_PROMPT = """
You are a brand new assistant
and we are demonstrating your instruction following capabilities. Note that you are in a crowded room and there may be background noise.
"""

# strong_agent = LanguageAgent(model_src="openai", api_key="OPENAI_API_KEY")
agent = LanguageAgent(model_src="openai", api_key="mbodi-demo-1", model_kwargs={"base_url": "http://localhost:3389/v1"})

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
NOT_A_COMPLETE_INSTRUCTION = "Not a complete instruction..."
weak_agent = LanguageAgent(model_src="openai", api_key="mbodi-demo-1", model_kwargs={"base_url": "http://localhost:3389/v1"},
    context=SYSTEM_PROMPT)


console = Console()
print = console.print # noqa: A001
gprint = partial(console.print, style="bold green on white")
def aprint(*args, **kwargs):
    console.print("ACT: ", *args, style="bold blue", **kwargs)
yprint = partial(console.print, style="bold yellow")
WAITING_FOR_NEXT_INSTRUCTION = "Waiting for next instruction..."

PredInstrMode = Literal["predict", "repeat", "clear", "wait"]
ActMode = Literal["acting", "repeat", "clear", "wait"]
SpeakMode = Literal["speaking", "wait", "clear"]

def predict_instruction(text: str, mode: PredInstrMode | str, last_instruction: str) -> Iterator[Tuple[str, PredInstrMode]]:  # noqa: UP006
    if mode == "clear" or not text or not text.strip():
        return "", "wait"
    if mode == "repeat":
        return last_instruction, mode

    if text == NOT_A_COMPLETE_INSTRUCTION:
        msg = "Instruction is not a complete instruction."
        raise ValueError(msg)

    yprint(f"Text: {text}, Mode Predict: {mode}, Last instruction: {last_instruction}")

    weak_agent.forget(everything=True)
    full_instruction = DETERMINE_INSTRUCTION_PROMPT + "\n" + text
    if weak_agent.act(instruction=full_instruction, model="astroworld", extra_body={"guided_choice": ["Yes", "No"]}) == "Yes":
        gprint(f"Instruction: is a complete instruction. Returning {text}")
        return text, "repeat"
    else:
        print(f"Instruction: {text} is not a complete instruction.")
        yield  NOT_A_COMPLETE_INSTRUCTION, "predict"

        yprint(weak_agent.act(instruction="Why wasn't it a complete instruction?", model="astroworld"))
        return NOT_A_COMPLETE_INSTRUCTION, "predict"


def act(instruction  : str, last_response: str, last_tps: str, mode: ActMode | str) -> Iterator[Tuple[str, str]]:  # noqa: UP006
    aprint(f"Instruction: {instruction}, last response: {last_response}, mode: {mode}")

    if mode == "clear" or not instruction or not instruction.strip():
        return "", last_tps, "wait"
    if mode == "wait":
        return "", last_tps, "wait"
    if mode == "repeat":
        return last_response, last_tps, "repeat"
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


def speak(text: str, mode: SpeakMode | str) -> Iterator[Tuple[bytes, str, bool]]:  # noqa: UP006
    """Generate and stream TTS audio using Coqui TTS with diarization."""
    console.print(f"SPEAK Text: {text}, Mode Speak: {mode}")
    if text and len(text.split()) < 3:
        return b"", "", "wait"

    if mode in ["clear", "wait"]:
        return b"", "", "wait"

    sentences = text.split(". ")
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    sentences = [*sentences, ""]
    for sentence in sentences:
        if sentence:
            audio_array = (np.array(tts.tts(sentence)) * 32767).astype(np.int16)
            pprint(f"Speaking: {sentence}")
            yield audio_array.tobytes(), sentence, "speaking"
        else:
            return "", "", "clear"
    print("Done speaking.")


def transition(instruction: str, response: str, instruction_mode: PredInstrMode,  act_mode: ActMode, speech_mode: SpeakMode) -> Iterator[Tuple[PredInstrMode, ActMode, SpeakMode]]:
    # If speech is completed, clear all modes
    if speech_mode == "clear":
        return "clear", "clear", "clear"

    # If no instruction yet, keep predicting
    if instruction == NOT_A_COMPLETE_INSTRUCTION:
        return "predict", "wait", "wait"

    # If we have an instruction but haven't acted on it
    if instruction and act_mode == "wait":
        return "repeat", "acting", "wait"


    # If we're acting and have a response, but haven't started speaking
    if act_mode == "acting" and response and speech_mode == "wait":
        return "repeat", "repeat", "speaking"


    # If we're speaking, keep the current modes
    if speech_mode == "speaking":
        return instruction_mode, act_mode, speech_mode


    # If we've finished speaking (speech_mode is "wait" after speaking)
    if speech_mode == "wait" and act_mode == "repeat" and instruction_mode == "repeat":
        return "predict", "wait", "wait"

    # Default case: maintain current modes
    return instruction_mode, act_mode, speech_mode

def create_gradio_demo(config: Config) -> gr.Blocks:
    base_url = "https://api.mbodi.ai/audio/v1"
    http_client = httpx.Client(base_url=base_url, timeout=TIMEOUT)
    openai_client = OpenAI(base_url=f"{base_url}", api_key="cant-be-empty")

    def handler(
        audio_source: str | tuple,
        model: str,
        task: Task,
        temperature: float,
        streaming: bool,
        stream: np.ndarray | None =None,
    ) -> Generator[np.ndarray, str, str, bool]:
        endpoint = TRANSLATION_ENDPOINT if task == Task.TRANSLATE else TRANSCRIPTION_ENDPOINT

        tic = time()
        total_tokens = 0

        if streaming:
            if not audio_source:
                return stream, ""
            sr, y = audio_source
            y = y.astype(np.float32)
            y = y.mean(axis=1) if y.ndim > 1 else y
            try:
                y /= np.max(np.abs(y))
            except Exception as e:
                logger.exception("Error normalizing audio: %s", traceback.format_exc())
                return np.array([]), "", "Error normalizing audio.", True
            stream = np.concatenate([stream, y]) if stream is not None else y
            if len(stream) < 16000:
                return stream, ""
            previous_transcription = ""
            for transcription in streaming_audio_task(stream, sr, endpoint, temperature, model):
                if previous_transcription.lower().strip().endswith(transcription.lower().strip()):
                    print(f"Skipping repeated transcription: {transcription}")
                    continue
                total_tokens = len(previous_transcription.split())
                elapsed_time = time() - tic
                tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
                previous_transcription += transcription
                yield stream, previous_transcription, f"STT tok/sec: {tokens_per_sec:.4f}", False

            return stream, previous_transcription, "Done speaking.", True

        else:
            result = ""
            result = audio_task(audio_source, endpoint, temperature, model)
            elapsed_time = time() - tic
            total_tokens = len(result.split())
            tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
            yield stream, result, f"STT tok/sec: {tokens_per_sec:.4f}", False
            return stream, result, "Done speaking.", True

    def streaming_audio_task(
        data: np.ndarray, sr: int, endpoint: str, temperature: float, model: str
    ) -> Generator[str, None, None]:
        kwargs = {
            "files": {"file": ("audio.wav", audio_to_bytes(sr, data), "audio/wav")},
            "data": {
                "response_format": "text",
                "temperature": temperature,
                "model": model,
                "stream": True,
            },
        }
        try:
            with connect_sse(http_client, "POST", endpoint, **kwargs) as event_source:
                for event in event_source.iter_sse():
                    yield event.data
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
            yield "Error streaming audio."


    def update_model_dropdown() -> gr.Dropdown:
        models = openai_client.models.list().data
        model_names: list[str] = [model.id for model in models]
        recommended_models = {model for model in model_names if model.startswith("Systran")}
        other_models = [model for model in model_names if model not in recommended_models]
        model_names = [*list(recommended_models), *other_models]
        return gr.Dropdown(
            choices=model_names,
            label="Model",
            value=config.whisper.model,
        )

    def update_audio_out(is_checked: bool):
        return gr.Audio(streaming=True, autoplay=True, label="Output", visible=is_checked, render=is_checked)

    def update_visibility(is_checked: bool):
        return (
            gr.Text(label="Instruction", visible=is_checked, render=is_checked),
            gr.Textbox(label="Response", placeholder="", visible=is_checked, render=is_checked),
            gr.Textbox(label="TPS", placeholder="", visible=is_checked, render=is_checked),
            gr.Slider(minimum=2, maximum=30, step=1, label="Memory", value=15, visible=is_checked, render=is_checked),
        )

    def clear_everything():
        return "clear", "clear", "clear"

    with gr.Blocks(
        theme=THEME, css=CUSTOM_CSS,
        title="Personal Assistant",
    ) as demo:
        audio = gr.Audio(label="Audio Input", type="numpy", sources=["microphone"], streaming=True, interactive=True, render=False)
        audio_out = gr.Audio(streaming=True, autoplay=True, label="Output", visible=False, render=False)

        transcription = gr.Textbox(label="Transcription", placeholder=PLACE_HOLDER, render=False)
        transcription_tps = gr.Textbox(label="TPS", placeholder="No data yet...", render=False)


        instruction = gr.Text(label="Instruction", visible=False, value="", render=False)
        response = gr.Textbox(label="Response", placeholder="", visible=False, render=False)
        response_tps = gr.Textbox(label="TPS", placeholder="No data yet...", visible=False, render=False)

        spoken_text = gr.Textbox(label="Spoken Text", placeholder="", visible=False)

        pred_instr_mode = gr.Textbox(value="predict",label="Predict Instruction Mode", placeholder="wait", render=True, visible=False)
        act_mode = gr.Textbox(value="wait", label="Act Mode", placeholder="wait", render=True, visible=False)
        speak_mode = gr.Textbox(value="wait",label="Speak Mode", placeholder="wait", render=True, visible=False)


        clear_button = gr.Button(value="Clear", render=False)
        memory = gr.Slider(minimum=2, maximum=30, step=1, label="Memory", value=15, visible=False, render=False)
        should_instruct = gr.Checkbox(label="Instruct", value=False, visible=True, render=False)
        should_speak = gr.Checkbox(label="Speak", value=False, visible=True, render=False)
        should_stream = gr.Checkbox(label="Stream", value=True, render=False)

        model_dropdown = gr.Dropdown(
                    choices=[config.whisper.model],
                    label="Model",
                    value=config.whisper.model,
                    render=False
                )
        task_dropdown = gr.Dropdown(
            choices=[task.value for task in Task],
            label="Task or System Prompt",
            value=Task.TRANSCRIBE,
            allow_custom_value=True,
            render=False
        )
        temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.0, render=False)
        stream = gr.State([])

        with gr.Row():
            temperature_slider.render()
            should_stream.render()
            with gr.Column():
                model_dropdown.render()
                task_dropdown.render()
                audio.render()
                audio_out.render()
                clear_button.render()
                should_instruct.render()
                should_speak.render()
            with gr.Column():
                transcription.render()
                instruction.render()
                memory.render()


        should_instruct.change(update_visibility, inputs=[should_instruct], outputs=[
            instruction, response, response_tps, memory
        ])
        should_speak.change(update_audio_out, inputs=[should_speak], outputs=[audio_out])
        gr.on([clear_button.click], clear_everything, inputs=None, outputs=[pred_instr_mode, act_mode, speak_mode])

        timer = gr.Timer(value=0.1, render=True)
        gr.on([transcription.change], 
        predict_instruction, 
                inputs=[transcription, pred_instr_mode,  instruction],
                outputs=[instruction, pred_instr_mode], trigger_mode="always_last"
        ).then(transition,
            inputs=[instruction, response, pred_instr_mode, act_mode, speak_mode],
            outputs=[pred_instr_mode, act_mode, speak_mode],
        )

        gr.on([timer.tick],
            predict_instruction,
                inputs=[transcription, pred_instr_mode, instruction],
                outputs=[instruction, pred_instr_mode],
                trigger_mode="always_last"
        ).then(
             transition,
                inputs=[instruction, response, pred_instr_mode, act_mode, speak_mode],
                outputs=[pred_instr_mode, act_mode, speak_mode],
        ).then(
             act,
                inputs=[instruction,response, response_tps,  act_mode],
                 outputs=[response, response_tps],
                 trigger_mode="always_last"
             ).then(
                transition,
                inputs=[instruction, response, pred_instr_mode, act_mode, speak_mode],
                outputs=[pred_instr_mode, act_mode, speak_mode],
            ).then(
                speak,
                inputs=[response, speak_mode],
                outputs=[audio_out, spoken_text, speak_mode]
             )
        demo.load(update_model_dropdown, inputs=None, outputs=model_dropdown)
        # audio_out.stream(lambda x,y: (x, y), inputs=[audio_out], outputs=[audio_out])
        audio.stream(handler, 
            inputs=[audio, model_dropdown, task_dropdown, temperature_slider, should_stream, stream],
            outputs=[stream, transcription, transcription_tps, pred_instr_mode])

    return demo



demo = create_gradio_demo(Config())

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    import sys
    import logging
    debug = sys.argv[-1] if len(sys.argv) > 1 else "INFO"
    logging.basicConfig(level=debug)
    logger = logging.getLogger(" ")
    logger.setLevel(logging.DEBUG)
    from rich.logging import RichHandler
    logger.addHandler(RichHandler())
    demo.queue().launch( server_name="0.0.0.0", share=False, show_error=True, debug=True, root_path="/instruct", server_port=7861)
