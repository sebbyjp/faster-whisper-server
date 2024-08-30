from collections.abc import Generator
from multiprocessing.managers import BaseManager, BaseProxy
import os
from pathlib import Path
from time import time
import traceback
from typing import Any

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
import httpx
from httpx_sse import connect_sse
from lager import log
from mbodied.agents.config import AgentConfig
import numpy as np
from pydantic import BaseModel, Field, SecretStr
from pydantic.json_schema import JsonSchemaValue
from pydantic_settings import BaseSettings, SettingsConfigDict
from xtyping import Callable, Dict, Iterable, Tuple

from faster_whisper_server.processing import audio_to_bytes

audio_state = {
    "stream": "",
    "model": "tts_models/multilingual/multi-dataset/xtts_v2",
    "temperature": 0.0,
}

class TaskConfig(BaseSettings):
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


from threading import Lock
lock = Lock()

def get_audio_state():
    with lock:
        return audio_state.copy()

def update_audio_state(new_state):
    with lock:
        audio_state.update(new_state)

audio_settings = TaskConfig()
settings_lock = Lock()
def get_audio_settings():
    with settings_lock:
        return audio_settings.copy()
def update_audio_settings(new_settings):
    with settings_lock:
        audio_settings.update(new_settings)
# class AgentConfig(BaseAgentConfig):
#     model_config = SettingsConfigDict(cli_parse_args=True, env_file=os.getenv("ENV_FILE", ".env"))
#     base_url: str = "https://api.mbodi.ai/v1"
#     auth_token: str = "mbodi-demo-1"
#     completion: CompletionConfig = Field(default_factory=CompletionConfig)
#     guidance: Guidance = Field(default_factory=Guidance)
#     sub_agents: list["AgentConfig"] | None = Field(default=None)
#     state: State = Field(default_factory=State)


WEBSOCKET_URI = "http://localhost:7543/v1/audio/transcriptions"





def map_language(language: str) -> str:
    if language == "en":
        return "English"
    if language == "ru":
        return "Russian"
    if language == "es":
        return "Spanish"
    if language == "fr":
        return "French"
    if language == "de":
        return "German"
    if language == "it":
        return "Italian"
    if language == "pt":
        return "Portuguese"
    return language




def stream_whisper(
    data: np.ndarray,
    sr: int,
    temperature: float,
    model: str,
    http_client: httpx.Client,
    endpoint: str = WEBSOCKET_URI,
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
        with connect_sse(http_client, "POST", endpoint, **kwargs) as event_source:
            for event in event_source.iter_sse():
                yield event.data
    except Exception as e:
        log.error(f"Error streaming audio: {e}")
        yield "Error streaming audio."


def handle_audio_file(
    file_path: str,
    state: Dict,
    endpoint: str,
    temperature: float,
    model: str,
    http_client: httpx.Client,
) -> Tuple[Dict, str, str]:
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


def normalize_audio(audio_source: tuple[int, np.ndarray]) -> np.ndarray:
    if not audio_source:
        return  "", ""
    sr, y = audio_source
    y = y.astype(np.float32)
    y = y.mean(axis=1) if y.ndim > 1 else y
    try:
        y /= np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1
        return sr, y
    except Exception:
        log.exception("Error normalizing audio: %s", traceback.format_exc())
        return sr, ""


def handle_audio_stream(
    audio_source: Tuple[int, np.ndarray] | None,
    temperature: float,
    http_client: httpx.Client,
    audio_state: Dict[str, Any],
    audio_settings: TaskConfig,
) -> Generator[Tuple[Dict, str, str], None, None]:
    """Handle audio data for transcription or translation tasks."""
    print(f"audio state: {audio_state}")

    stream = audio_state["stream"]
    if len(stream) < 16000:
        audio_state["stream"] = stream
        return audio_state, "", ""
    sr, y = normalize_audio(audio_source)
    previous_transcription = ""
    model = audio_state["model"]
    tic = time()
    for transcription in stream_whisper(stream, sr, endpoint=audio_settings.TRANSCRIPTION_ENDPOINT, temperature=temperature, model=model, http_client=http_client):
        if previous_transcription.lower().strip().endswith(transcription.lower().strip()):
            print(f"Skipping repeated transcription: {transcription}")
            continue
        total_tokens = len(previous_transcription.split())
        elapsed_time = time() - tic
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        previous_transcription += transcription
        print(f"Transcription: {previous_transcription}, State: {audio_state.update({'stream': stream})}")
        audio_state["stream"] = stream
        yield audio_state, previous_transcription, f"STT tok/sec: {tokens_per_sec:.4f}"
    print(f"Transcription: {previous_transcription}, State: {audio_state}")
    return audio_state, previous_transcription, f"STT tok/sec: {tokens_per_sec:.4f}"
