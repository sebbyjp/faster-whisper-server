import asyncio
from collections.abc import AsyncGenerator
from functools import partial
import logging
from time import time
from typing import Literal

import gradio as gr
import httpx
import numpy as np
from openai import AsyncOpenAI
from rich.console import Console

# Constants
TIMEOUT = 30
THEME = gr.themes.Base()
CUSTOM_CSS = """
/* Add your custom CSS here */
"""
PLACE_HOLDER = "No data yet..."
NOT_A_COMPLETE_INSTRUCTION = "Not a complete instruction..."
DETERMINE_INSTRUCTION_PROMPT = "Determine if the following is a complete instruction:"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Type aliases
PredInstrMode = Literal["predict", "repeat", "clear", "wait"]
ActMode = Literal["acting", "repeat", "clear", "wait"]
SpeakMode = Literal["speaking", "wait", "clear"]
Task = Literal["TRANSCRIBE", "TRANSLATE"]

# Console setup
console = Console()
print = console.print
gprint = partial(console.print, style="bold green on white")
aprint = partial(console.print, style="bold blue")
yprint = partial(console.print, style="bold yellow")

# Assuming these are defined elsewhere in your code
weak_agent = LanguageAgent(model_src="openai", api_key="mbodi-demo-1", model_kwargs={"base_url": "http://localhost:3389/v1"},
    context=SYSTEM_PROMPT)
agent = LanguageAgent(model_src="openai", api_key="mbodi-demo-1", model_kwargs={"base_url": "http://localhost:3389/v1"},
    context=SYSTEM_PROMPT)

async def predict_instruction(text: str | None, mode: PredInstrMode, last_instruction: str) -> AsyncGenerator[tuple[str, PredInstrMode], None]:
    if mode == "clear" or not text or not text.strip():
        yield "", "wait"
        return
    if mode == "repeat":
        yield last_instruction, mode
        return

    if text == NOT_A_COMPLETE_INSTRUCTION:
        msg = "Instruction is not a complete instruction."
        logger.warning(msg)
        yield NOT_A_COMPLETE_INSTRUCTION, "predict"
        return

    yprint(f"Text: {text}, Mode Predict: {mode}, Last instruction: {last_instruction}")

    weak_agent.forget(everything=True)
    full_instruction = DETERMINE_INSTRUCTION_PROMPT + "\n" + text
    if await weak_agent.act(instruction=full_instruction, model="astroworld", extra_body={"guided_choice": ["Yes", "No"]}) == "Yes":
        gprint(f"Instruction: is a complete instruction. Returning {text}")
        yield text, "repeat"
    else:
        print(f"Instruction: {text} is not a complete instruction.")
        yield NOT_A_COMPLETE_INSTRUCTION, "predict"

        yprint(await weak_agent.act(instruction="Why wasn't it a complete instruction?", model="astroworld"))
        yield NOT_A_COMPLETE_INSTRUCTION, "predict"

async def act(instruction: str | None, last_response: str | None, last_tps: str, mode: ActMode) -> AsyncGenerator[tuple[str, str, ActMode], None]:
    aprint(f"Instruction: {instruction}, last response: {last_response}, mode: {mode}")

    if mode == "clear" or not instruction or not instruction.strip():
        yield "", last_tps, "wait"
        return
    if mode == "wait":
        yield "", last_tps, "wait"
        return
    if mode == "repeat":
        yield last_response or "", last_tps, "repeat"
        return
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
    async for text in agent.act_and_stream(instruction=instruction, model="astroworld"):
        total_tokens = len(response.split())
        elapsed_time = time() - tic
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        response += text
        aprint(f"Response: {response}")
        if response == last_response:
            yield response, f"TPS: {tokens_per_sec:.4f}", "repeat"
            return
        yield response, f"TPS: {tokens_per_sec:.4f}", "acting"

async def speak(text: str | None, mode: SpeakMode) -> AsyncGenerator[tuple[bytes, str, SpeakMode], None]:
    console.print(f"SPEAK Text: {text}, Mode Speak: {mode}")
    if not text or len(text.split()) < 3:
        yield b"", "", "wait"
        return

    if mode in ["clear", "wait"]:
        yield b"", "", "wait"
        return

    sentences = text.split(". ")
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    sentences = [*sentences, ""]
    for sentence in sentences:
        if sentence:
            audio_array = (np.array(tts.tts(sentence)) * 32767).astype(np.int16)
            pprint(f"Speaking: {sentence}")
            yield audio_array.tobytes(), sentence, "speaking"
        else:
            yield b"", "", "clear"
    print("Done speaking.")

def transition(instruction: str | None, response: str | None, instruction_mode: PredInstrMode, act_mode: ActMode, speech_mode: SpeakMode) -> tuple[PredInstrMode, ActMode, SpeakMode]:
    if speech_mode == "clear":
        return "clear", "clear", "clear"

    if not instruction or instruction == NOT_A_COMPLETE_INSTRUCTION:
        return "predict", "wait", "wait"

    if instruction and act_mode == "wait":
        return "repeat", "acting", "wait"

    if act_mode == "acting" and response and speech_mode == "wait":
        return "repeat", "repeat", "speaking"

    if speech_mode == "speaking":
        return instruction_mode, act_mode, speech_mode

    if speech_mode == "wait" and act_mode == "repeat" and instruction_mode == "repeat":
        return "predict", "wait", "wait"

    return instruction_mode, act_mode, speech_mode

class State:
    def __init__(self) -> None:
        self.instruction: str | None = ""
        self.response: str | None = ""
        self.transcription: str | None = ""
        self.pred_instr_mode: PredInstrMode = "predict"
        self.act_mode: ActMode = "wait"
        self.speak_mode: SpeakMode = "wait"

async def process_state(state: State) -> tuple[str | None, str | None, str, PredInstrMode, ActMode, SpeakMode, bytes, str]:
    new_instruction, new_pred_mode = "", state.pred_instr_mode
    async for instr, mode in predict_instruction(state.transcription, state.pred_instr_mode, state.instruction):
        new_instruction, new_pred_mode = instr, mode

    new_pred_mode, new_act_mode, new_speak_mode = transition(new_instruction, state.response, new_pred_mode, state.act_mode, state.speak_mode)

    new_response, new_tps = state.response, ""
    audio_chunk, spoken_text = b"", ""

    if new_act_mode == "acting":
        async for resp, tps, mode in act(new_instruction, state.response, "", new_act_mode):
            new_response, new_tps, new_act_mode = resp, tps, mode

    if new_speak_mode == "speaking":
        async for chunk, text, mode in speak(new_response, new_speak_mode):
            audio_chunk, spoken_text, new_speak_mode = chunk, text, mode

    return new_instruction, new_response, new_tps, new_pred_mode, new_act_mode, new_speak_mode, audio_chunk, spoken_text

def create_gradio_demo(config: Config) -> gr.Blocks:
    base_url = "https://api.mbodi.ai/audio/v1"
    httpx.Client(base_url=base_url, timeout=TIMEOUT)
    AsyncOpenAI(base_url=f"{base_url}", api_key="cant-be-empty")
    state = State()

    async def handler(
        audio_source: tuple[int, np.ndarray] | None,
        model: str,
        task: Task,
        temperature: float,
        streaming: bool,
        stream: np.ndarray
    ) -> tuple[np.ndarray, str, str, bool]:
        endpoint = TRANSLATION_ENDPOINT if task == Task.TRANSLATE else TRANSCRIPTION_ENDPOINT

        if not audio_source:
            return stream, "", "", False

        sr, y = audio_source
        y = y.astype(np.float32)
        y = y.mean(axis=1) if y.ndim > 1 else y
        try:
            y /= np.max(np.abs(y))
        except Exception as e:
            logger.exception(f"Error normalizing audio: {e!s}")
            return np.array([]), "", "Error normalizing audio.", True

        stream = np.concatenate([stream, y]) if stream is not None else y
        if len(stream) < 16000:
            return stream, "", "", False

        previous_transcription = ""
        async for transcription in streaming_audio_task(stream, sr, endpoint, temperature, model):
            if previous_transcription.lower().strip().endswith(transcription.lower().strip()):
                print(f"Skipping repeated transcription: {transcription}")
                continue
            previous_transcription += transcription
            state.transcription = previous_transcription
            yield stream, previous_transcription, "Transcription in progress", False

        return stream, previous_transcription, "Transcription complete", True

    async def update_state():
        while True:
            new_instruction, new_response, new_tps, new_pred_mode, new_act_mode, new_speak_mode, audio_chunk, spoken_text = await process_state(state)
            state.instruction = new_instruction
            state.response = new_response
            state.pred_instr_mode = new_pred_mode
            state.act_mode = new_act_mode
            state.speak_mode = new_speak_mode

            yield {
                instruction: new_instruction,
                response: new_response,
                response_tps: new_tps,
                pred_instr_mode: new_pred_mode,
                act_mode: new_act_mode,
                speak_mode: new_speak_mode,
                audio_out: audio_chunk,
                spoken_text: spoken_text
            }
            await asyncio.sleep(0.1)

    with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="Personal Assistant") as demo:
        audio = gr.Audio(label="Audio Input", type="numpy", sources=["microphone"], streaming=True)
        audio_out = gr.Audio(label="Output", streaming=True, autoplay=True)

        transcription = gr.Textbox(label="Transcription", placeholder=PLACE_HOLDER)
        instruction = gr.Textbox(label="Instruction", placeholder=PLACE_HOLDER)
        response = gr.Textbox(label="Response", placeholder=PLACE_HOLDER)
        response_tps = gr.Textbox(label="Response TPS", placeholder=PLACE_HOLDER)

        pred_instr_mode = gr.Textbox(label="Predict Instruction Mode", value="predict")
        act_mode = gr.Textbox(label="Act Mode", value="wait")
        speak_mode = gr.Textbox(label="Speak Mode", value="wait")

        spoken_text = gr.Textbox(label="Spoken Text", placeholder=PLACE_HOLDER)

        model_dropdown = gr.Dropdown(choices=[config.whisper.model], label="Model", value=config.whisper.model)
        task_dropdown = gr.Dropdown(choices=[task.value for task in Task], label="Task", value=Task.TRANSCRIBE)
        temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.0)
        should_stream = gr.Checkbox(label="Stream", value=True)

        stream = gr.State(np.array([]))

        audio.stream(
            handler,
            inputs=[audio, model_dropdown, task_dropdown, temperature_slider, should_stream, stream],
            outputs=[stream, transcription, response_tps, pred_instr_mode]
        )

        demo.load(lambda: gr.Button.update(interactive=True))
        demo.queue()
        demo.add_event(update_state, None, [instruction, response, response_tps, pred_instr_mode, act_mode, speak_mode, audio_out, spoken_text], every=0.1)

    return demo

if __name__ == "__main__":
    config = Config()  # Assuming Config is defined elsewhere in your code
    demo = create_gradio_demo(config)
    demo.launch()
