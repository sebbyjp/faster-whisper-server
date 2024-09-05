from collections.abc import Generator, Iterator
from functools import partial
import os
import re
import threading
from time import time
import traceback
from typing import Any, Dict, Literal, Tuple
import soundfile as sf
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
from faster_whisper_server.audio_task import State, TaskConfig, handle_audio_stream
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

_agent_state: Dict[str, Any] = {
    "pred_instr_mode": "predict",
    "act_mode": "wait",
    "speak_mode": "wait",
    "transcription": "",
    "instruction": "",
    "response": "",
    "spoken": "",
    "audio_array": np.array([0], dtype=np.int16),
}

_audio_state: Dict[str, Any] = {
    "stream": np.array([]),
    "model": "Systran/faster-distil-whisper-large-v3",
    "temperature": 0.0,
    "endpoint": "/audio/transcriptions"
}

# Lock for thread-safe access to global state
state_lock = threading.Lock()
audio_lock = threading.Lock()
def get_state() -> Dict[str, Any]:
    with state_lock:
        return _agent_state

def update_state(updates: Dict[str, Any]) -> None:
    with state_lock:
        _agent_state.update(updates)

def get_audio() -> Dict[str, Any]:
    with audio_lock:
        return _audio_state

def update_audio(updates: Dict[str, Any]) -> None:
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


# class AgentState(State):
#     pred_instr_mode: Literal["predict", "repeat", "clear", "wait"]
#     act_mode: Literal["acting", "repeat", "clear", "wait"]
#     speak_mode: Literal["speaking", "wait", "clear"]
#     transcription: str
#     instruction: str




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
    if mode not in ["acting", "wait"]:
        console.print(f"ERROR: Invalid mode: {mode}", style="bold red")
        raise ValueError(f"Invalid mode: {mode}")

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
            update_state({"act_mode": "repeat", "response": response})
            return response, f"TPS: {tokens_per_sec:.4f}", get_state()
        update_state({"act_mode": "acting","response": response, "speak_mode": "speaking"})
        yield response, f"TPS: {tokens_per_sec:.4f}", get_state()
    update_state({"act_mode": "repeat","response": response, "speak_mode": "wait"})
    return response, f"TPS: {tokens_per_sec:.4f}", get_state()

def speak(text: str, state: Dict, speaker: str, language: str) -> Iterator[Tuple[bytes, str, Dict]]:  # noqa: UP006
    """Generate and stream TTS audio using Coqui TTS with diarization."""
    state = get_state()
    print(f"SPEAK: state: {state}")
    text = state["response"]
    mode = state["speak_mode"]
    sr = tts.synthesizer.output_sample_rate

    if text and len(text.split()) < 3 and not (text.endswith(".") or text.endswith("?") or text.endswith("!")):
        state["speak_mode"] = "wait"
        return b"", state

    if mode in ("clear", "wait"):
        state["speak_mode"] = "wait"
        return b"",  state

    sentences = [sentence.strip() for sentence in re.split(r'[.?!]', text) if sentence.strip()]
    # sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    sentences = [*sentences, ""]
    spoken = state["spoken"]
    audio_array = state["audio_array"]
    for sentence in sentences:
        if sentence and not (spoken and spoken.endswith(sentence)):
            audio_array = (np.array(tts.tts(sentence, speaker=speaker, language=language, split_sentences=False)) * 32767).astype(
                np.int16
            )
            console.print(f"SPEAK Text: {text}, Mode Speak: {mode}", style="bold white on blue")
            state["speak_mode"] = "speaking"
            spoken += sentence
            update_state({"speak_mode": "speaking", "spoken": spoken, "audio_array": audio_array, "act_mode": "repeat"})
            sf.write("out.wav", audio_array,sr)
        else:
            continue
    state["speak_mode"] = "finished"
    update_state({"speak_mode": "finished"})
    console.print("Done speaking.", style="bold white on blue")
    return (sr, audio_array), state



def process_state(new_transcription: str, new_instruction: str, response: str, state: Dict) -> Dict:
    print(f"PROCESS: Current state: {state}, new transcription: {new_transcription}, new instruction: {new_instruction}, response: {response}")
    state = get_state()
    new_instruction = state["instruction"]
    # If any mode is "clear", reset all modes
    if any(state.get(mode) == "clear" for mode in ["pred_instr_mode", "act_mode", "speak_mode"]):
        update_state(
            {
                "pred_instr_mode": "predict",
                "act_mode": "wait",
                "speak_mode": "wait",
                "transcription": "",
                "instruction": "",
            }
        )
        return {
            "pred_instr_mode": "predict",
            "act_mode": "wait",
            "speak_mode": "wait",
            "transcription": "",
            "instruction": "",
        }, "", "", ""

    # If no complete instruction yet
    # Wait for more transcription before predicting
    if not new_transcription or not new_transcription.strip() or new_transcription == "Not enough audio yet.":
        yprint(f"Waiting for more audio. Current transcription: {new_transcription}")

        update_state(
            {
                "pred_instr_mode": "wait",
                "act_mode": "wait",
                "speak_mode": "wait",
                "transcription": new_transcription,
                "instruction": new_instruction,
            }
        )
        return {
            "pred_instr_mode": "predict",
            "act_mode": "wait",
            "speak_mode": "wait",
            "transcription": new_transcription,
            "instruction": new_instruction,
        }, new_transcription, new_instruction, response

    # If we have an incomplete instruction
    # Continue or start predicting
    if new_transcription and (not new_instruction or new_instruction == NOT_A_COMPLETE_INSTRUCTION):
        yprint(f"Predicting instruction for: {new_transcription}")
        update_state(
            {
                "pred_instr_mode": "predict",
                "act_mode": "wait",
                "speak_mode": "wait",
                "transcription": new_transcription,
                "instruction": new_instruction,
            }
        )
        return {
            "pred_instr_mode": "predict",
            "act_mode": "wait",
            "speak_mode": "wait",
            "transcription": new_transcription,
            "instruction": new_transcription,
        }, new_transcription, new_instruction, response

    # If we have an instruction but haven't acted on it
    # Repeat transcription and instruction
    if new_instruction and state["act_mode"] == "wait":
        yprint(f"Acting on instruction: {new_instruction}")
        update_state(
            {
                "pred_instr_mode": "repeat",
                "act_mode": "acting",
                "speak_mode": "wait",
                "transcription":state["transcription"],
                "instruction":state["instruction"],
                "response": response,
            }
        )
        return ({
            "pred_instr_mode": "repeat",
            "act_mode": "acting",
            "speak_mode": "wait",
            "transcription":state["transcription"],
            "instruction":state["instruction"],
        }, state["transcription"], state["instruction"], response)

    # If we're acting and have a response, but haven't started speaking
    if state["act_mode"] == "repeat" and response and state["speak_mode"] in ("wait", "speaking"):
        yprint(f"Speaking response: {response}")
        update_state(
            {
                "pred_instr_mode": "repeat",
                "act_mode": "repeat",
                "speak_mode": "speaking",
                "transcription": get_state()["transcription"],
                "instruction": get_state()["instruction"],
                "response": response,
            }
        )
        return ({
            "pred_instr_mode": "repeat",
            "act_mode": "repeat",
            "speak_mode": "speaking",
            "transcription": get_state()["transcription"],
            "instruction": get_state()["instruction"],
            "response": response,
        }, state["transcription"], state["instruction"], response)
    # If we've finished speaking
    if (state["spoken"] and state["spoken"] == state["response"]) or state["speak_mode"] == "finished":
        yprint("Resetting state.")
        update_state(
            {
                "pred_instr_mode": "predict",
                "act_mode": "wait",
                "speak_mode": "wait",
                "transcription": "",
                "instruction": "",
                "response": "",
                "spoken": "",
                "audio_array": np.array([0], dtype=np.int16),
            }
        )
        update_audio({"stream": np.array([])})
        return state, new_transcription, new_instruction, response
    # Default case: maintain current state
    return state, state["transcription"], state["instruction"], state["response"]

def create_gradio_demo(config: AudioConfig, task_config: TaskConfig) -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue=mbodi_color,
            secondary_hue="stone",
        ),
        title="Personal Assistant",
        delete_cache=[0,0]
    ) as demo:
        # audio_out = gr.Audio(streaming=True, autoplay=True, label="Output", visible=False, render=False)
        # temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.0, render=False)

        # state = gr.State(
        #     State(pred_instr_mode="predict", act_mode="wait", speak_mode="wait", transcription="", instruction="")
        # )

        # task_config = gr.State(task, render=False)

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

    

        # clear_button = gr.Button(value="Clear", render=False)

        # should_instruct = gr.Checkbox(label="Instruct", value=False, visible=True, render=False)
        # should_speak = gr.Checkbox(label="Speak", value=False, visible=True, render=False)
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
            agent_state = gr.State(
                {
                    "pred_instr_mode": "predict",
                    "act_mode": "wait",
                    "speak_mode": "wait",
                    "transcription": "",
                    "instruction": "",
                }
            )
            def stream_audio(
                audio: Tuple[int, np.ndarray],
                audio_state: Dict,
                model: str,
            ) -> Iterator[tuple[str, str]]:
                # print(f"THIS IS THE AUDIO STATE: {audio_state}")
                audio_state = get_audio()
                audio_state["model"] = model
                for state, transcription, transcription_tps in handle_audio_stream(
                    audio,
                    audio_state,
                    0.0,
                    http_client,
                ):
                    update_audio(state)
                    yield get_audio(), transcription, transcription_tps

            audio.stream(
                fn=stream_audio,
                inputs=[
                    audio,
                    audio_state,
                    model_dropdown,
                ],
                outputs=[audio_state, transcription, transcription_tps]
            ).then(
                process_state,
                inputs=[transcription,instruction, response, agent_state],
                outputs=[agent_state, transcription, instruction, response],
            )
            transcription.change(
                predict_instruction,
                inputs=[transcription, instruction, agent_state],
                outputs=[instruction, agent_state],
            ).then(
                process_state,
                inputs=[transcription,instruction, response, agent_state],
                outputs=[agent_state, transcription, instruction, response],
            )
            is_speaking = gr.Checkbox(label="Speak", value=False, render=True, visible=False)
            instruction.change(
                act,
                inputs=[instruction, response, response_tps, agent_state],
                outputs=[response, response_tps, agent_state],
            )
            # response.change(
            #     lambda: get_state()["speak_mode"] == "speaking",
            #     inputs=None,
            #     outputs=[is_speaking],
            # )
            response.change(
                speak,
                inputs=[response, agent_state, first_speaker_name, first_speaker_language],
                outputs=[audio_out, agent_state],
            )
        # audio_out.stream(speak, inputs=[response, state], outputs=[audio_out, state], trigger_mode="always_last")
        demo.load(update_model_dropdown, inputs=None, outputs=[model_dropdown])
        return demo


demo = create_gradio_demo(AudioConfig(), TaskConfig())

if __name__ == "__main__":
    # debug = sys.argv[-1] if len(sys.argv) > 1 else "INFO"
    from rich.logging import RichHandler

    log.add(RichHandler(), level="DEBUG")
    demo.launch(
        server_name="0.0.0.0", share=False, show_error=True, debug=True, root_path="/instruct", server_port=7861
    )



  #         fn=update_dropdowns,
        #         inputs=[
        #             model_dropdown,
        #             first_speaker_language,
        #             second_speaker_language,
        #             first_speaker_name,
        #             second_speaker_name,
        #             task_config,
        #             audio_state,
        #         ],
        #         outputs=[
        #             model_dropdown,
        #             first_speaker_language,
        #             second_speaker_language,
        #             first_speaker_name,
        #             second_speaker_name,
        #             task_config,
        #             audio_state,
        #         ],
        #     )
        # should_speak.render()
        # instruction.render()
        # def audio_stream(audio, audio_state, temperature) -> Iterator[Tuple[AudioState, str, str]]:
        #     print(f"THIS IS THE AUDIO STATE: {audio}")
        #     yield from handle_audio_stream(audio, audio_state, config, temperature, http_client)

        # gr.on(
        #     audio.change,
        #     fn=audio_stream,
        #     inputs=[
        #         audio,
        #         audio_state,
        #         temperature_slider,
        #     ],
        #     outputs=[audio_state,transcription, transcription_tps],
        #     trigger_mode="always_last",
        # )
        # ).then(process_state, inputs=[instruction, response, state], outputs=[state], trigger_mode="always_last").then(
        #     predict_instruction,
        #     inputs=[transcription,instruction, state],
        #     outputs=[instruction, state],
        # )
        # timer = Timer(value=0.5, render=True)
        # gr.on([timer.tick], process_state,  inputs=[instruction, response, state], outputs=[state], trigger_mode="always_last").then(
        #     act,
        #     inputs=[instruction, response, response_tps, state],
        #     outputs=[response, response_tps, state],
        # ).then(process_state,  inputs=[instruction, response, state], outputs=[state], trigger_mode="always_last")

        # def update_audio_out(is_checked: bool):
        #     return gr.Audio(stream=speak, visible=is_checked, render=is_checked)

        # gr.on(
        #     [should_speak.change],
        #     fn=update_audio_out,
        #     inputs=[should_speak],
        #     outputs=[audio_out],
        # )
        # def update_visibility(is_checked: bool):
        #     return (
        #         gr.Text(label="Instruction", visible=is_checked, render=is_checked),
        #         gr.Textbox(label="Response", placeholder="", visible=is_checked, render=is_checked),
        #         gr.Textbox(label="TPS", placeholder="", visible=is_checked, render=is_checked),
        #     )

        # gr.on([should_instruct.change],
        #     inputs=[should_instruct],
        #     outputs=[instruction, response, response_tps],
        #     fn=update_visibility,
        # )

        # @gr.on(
        #     [clear_button.click],
        #     inputs=[clear_button],
        #     outputs=[state, audio_state, transcription, transcription_tps, instruction, response, response_tps],
        # )
        # def clear_everything(audio_state): # noqa
        #     return (
        #         State(pred_instr_mode="clear", act_mode="clear", speak_mode="clear", transcription="", instruction=""),
        #         AudioState(
        #             stream=np.array([]),
        #             model=audio_state["model"],
        #             transcription="",
        #             temperature=audio_state.get("temperature", 0.0),
        #         ),
        #         "",
        #         "",
        #         "",
        #         "",
        #         "",
        #     )