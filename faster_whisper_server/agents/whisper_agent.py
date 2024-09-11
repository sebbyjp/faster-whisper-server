from collections.abc import Generator
from functools import wraps
import logging
from pathlib import Path
from time import time
import traceback
from typing import Iterable

import gradio as gr
import httpx
from httpx_sse import connect_sse
from lager import log
import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from typing_extensions import TypedDict, override  # noqa

from faster_whisper_server.agents.agent import StatefulAgent
from faster_whisper_server.agents.config import AgentConfig, CompletionConfig, State, persist_maybe_clear
from faster_whisper_server.colors import mbodi_color
from faster_whisper_server.processing import audio_to_bytes

WEBSOCKET_URI="http://localhost:7543/v1/audio/transcriptions"

def normalize_audio(audio_source: tuple[int, NDArray], local_state: State, shared_state: State) -> tuple[State, str, str]:
    """Normalize audio to have a max amplitude of 1."""
    if local_state.check_clear(shared_state) or not audio_source:
        return ""

    sr, y = audio_source
    y = y.astype(np.float32)
    y = y.mean(axis=1) if y.ndim > 1 else y
    try:
        y /= (np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1)
        y = np.concatenate([local_state.get("stream", np.array([], dtype=np.float32)), y])
    except Exception as e:
        logging.exception("Error normalizing audio: %s", traceback.format_exc())

    return sr, y




 

whisper_config = AgentConfig(
    stream_config=CompletionConfig(
    response_modifier=persist_maybe_clear,
    prompt_modifier=normalize_audio,
    ),
    gradio_io=lambda:(
        gr.Audio(streaming=True, label="input", render=True),
        [gr.Textbox(label="Trancription", render=True), gr.Textbox(label="Tokens / Seconds", render=True)],
    ),
    base_url= "http://localhost:3389/v1",
    transcription_endpoint= "/audio/transcriptions",
    translation_endpoint = "/audio/translations",
    websocket_uri = "wss://api.mbodi.ai/audio/v1/transcriptions",
    temperature = 0.0,
)


def stream_whisper(
    data: np.ndarray,
    sr: int,
    endpoint: str, # noqa
    temperature: float,
    model: str,
    http_client: httpx.Client,
) -> Iterable[str]:
    """Stream audio data to the server and yield transcriptions."""
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
        with connect_sse(http_client, "POST", WEBSOCKET_URI, **kwargs) as event_source:
            for event in event_source.iter_sse():
                yield event.data
    except Exception as e: # noqa
        log.error(f"Error streaming audio: {e}")
        yield "Error streaming audio."


def handle_audio_file(
    file_path: str,
    state: dict,
    endpoint: str,
    temperature: float,
    model: str,
    http_client: httpx.Client,
) -> tuple[dict, str, str]:
    tic = time()
    http_client: httpx.Client
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
    result = response.text
    response.raise_for_status()
    elapsed_time = time() - tic
    total_tokens = len(result.split())
    tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
    return state, result, f"STT tok/sec: {tokens_per_sec:.4f}"

from rich.console import Console

console = Console(style="bold magenta")
class WhisperAgent(StatefulAgent):
    def __init__(self, config: AgentConfig, shared_state: State) -> None:
        super().__init__(config, shared_state)
        self.config: AgentConfig = config
        self.http_client = httpx.Client(base_url=config.base_url, timeout=60)
        self.temperature = config.temperature

    def handle_stream(self, audio_source: tuple[int, NDArray[np.float32]]) -> Generator[tuple[str, str], None, None]:
        """Handle audio data for transcription or translation tasks."""
        print(f"audio state: {self.local_state}")
        print(f"auido source: {audio_source}")
        endpoint = self.config.transcription_endpoint
        tic = time()
        total_tokens = 0

        # stream = audio_state["stream"]
        # stream = np.concatenate([stream, y]) if stream is not None else y
        # if len(stream) < 16000:
        #     audio_state["stream"] = stream
        #     return  audio_state, "", ""
        previous_transcription = ""
        model = self.config.model
        sr, stream = audio_source
        for transcription in stream_whisper(stream, sr, endpoint, self.config.temperature, model, self.http_client):
            total_tokens = len(previous_transcription.split())
            elapsed_time = time() - tic
            tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
            previous_transcription += transcription
            # print(f"Transcription: {previous_transcription}, State: {audio_state.update({'stream': stream})}")
            # audio_state["stream"] = stream
            console.print(f"transcription: {previous_transcription}")
            yield previous_transcription, f"STT tok/sec: {tokens_per_sec:.4f}"
        print(f"Transcription: {previous_transcription}, State: {self.local_state}")
        return previous_transcription, f"STT tok/sec: {tokens_per_sec:.4f}"

    def demo(self) -> gr.Interface:
        """Launch the agent."""
        inputs, outputs = self.config.gradio_io()
        return gr.Interface(
            self.stream,
            inputs=inputs,
            outputs=outputs,
            title="Whisper",
            description="Stream audio to the server for transcription.",
            live=True,
            theme=gr.themes.Soft(
            primary_hue=mbodi_color,
            secondary_hue="stone",
            ),
        )

if __name__ == "__main__":
    state = State(is_first=True, is_terminal=False)
    whisper_agent = WhisperAgent(whisper_config, shared_state=state)

    demo = whisper_agent.demo()
    with demo as demo:
        demo.launch(
        server_name="0.0.0.0", # noqa
        server_port=7861,
        share=False,
        debug=True,
        show_error=True,
        )






