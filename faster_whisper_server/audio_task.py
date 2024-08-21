from collections.abc import Generator
from pathlib import Path
from time import time
import traceback

import httpx
from httpx_sse import connect_sse
from lager import log
import numpy as np
from pydantic_settings import BaseSettings
from typing_extensions import TypedDict, override  # noqa
from xtyping import Any, Dict, Iterable, Tuple

from faster_whisper_server.processing import audio_to_bytes

WEBSOCKET_URI="http://localhost:7543/v1/audio/transcriptions"
class State(TypedDict):
    @override
    def update(self, key: str, value: Any | None = None) -> "State":
        self[key] = value
        return self

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
    FIRST_SPEAKER: str = "Luis Moray"
    SECOND_SPEAKER: str = "Sofia Hellen"
    FIRST_LANGUAGE: str = "en"
    SECOND_LANGUAGE: str = "es"
    GPU: str = "cuda:6"


def stream_whisper(
    data: np.ndarray,
    sr: int,
    endpoint: str,
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


def handle_audio_stream(
    audio_source: Tuple[int, np.ndarray] | None,
    audio_state: State,
    temperature: float,
    http_client: httpx.Client,
) -> Generator[Tuple[Dict, str, str], None, None]:
    """Handle audio data for transcription or translation tasks."""
    print(f"audio state: {audio_state}")
    endpoint = audio_state["endpoint"]
    tic = time()
    total_tokens = 0
    if not audio_source:
        return audio_state, "", ""
    sr, y = audio_source
    y = y.astype(np.float32)
    y = y.mean(axis=1) if y.ndim > 1 else y
    try:
        y /= np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1
    except Exception as e:
        log.exception("Error normalizing audio: %s", traceback.format_exc())
        return audio_state, "", ""
    stream = audio_state["stream"]
    stream = np.concatenate([stream, y]) if stream is not None else y
    if len(stream) < 16000:
        audio_state["stream"] = stream
        return  audio_state, "", ""
    previous_transcription = ""
    model = audio_state["model"]
    for transcription in stream_whisper(stream, sr, endpoint, temperature, model, http_client):
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
