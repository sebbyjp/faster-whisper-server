import gradio as gr
import numpy as np
from faststream import FastStream, Stream
from multiprocessing import Manager
from time import time
from typing import Dict, Generator, Tuple
from io import BytesIO
import soundfile as sf
import httpx
from httpx_sse import connect_sse

# Initialize FastStream app
app = FastStream()

# Configuration for HTTP client
class TaskConfig:
    agent_base_url: str = "http://localhost:3389/v1"
    agent_token: str = "mbodi-demo-1"

http_client = httpx.Client(base_url=TaskConfig.agent_base_url, headers={"Authorization": f"Bearer {TaskConfig.agent_token}"})

TRANSLATION_ENDPOINT = "/translate"
TRANSCRIPTION_ENDPOINT = "/transcribe"

# Helper function to convert audio to bytes
def audio_to_bytes(sr: int, data: np.ndarray) -> bytes:

    with BytesIO() as buf:
        sf.write(buf, data, sr, format='WAV')
        return buf.getvalue()

# Create a shared state using multiprocessing.Manager
manager = Manager()
shared_state = manager.dict({
    "stream": None,
    "transcription": "",
    "tokens_per_sec": 0.0,
})

# FastStream tasks
@app.task
async def streaming_audio_task(data: np.ndarray, sr: int, endpoint: str, temperature: float, model: str) -> Stream[str]:
    global http_client
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
        yield f"Error streaming audio: {str(e)}"

@app.task
async def handle_audio(audio_source, task, streaming, temperature, model):
    stream = shared_state.get("stream", None)
    endpoint = TRANSLATION_ENDPOINT if task == "translate" else TRANSCRIPTION_ENDPOINT

    tic = time()
    total_tokens = 0

    if streaming:
        if not audio_source:
            return
        
        sr, y = audio_source
        y = y.astype(np.float32)
        y = y.mean(axis=1) if y.ndim > 1 else y
        
        try:
            y /= np.max(np.abs(y))
        except Exception as e:
            shared_state["transcription"] = f"Error normalizing audio: {str(e)}"
            return
        
        stream = np.concatenate([stream, y]) if stream is not None else y
        if len(stream) < 16000:
            shared_state["stream"] = stream
            return
        
        previous_transcription = shared_state.get("transcription", "")
        async for transcription in streaming_audio_task(stream, sr, endpoint, temperature, model):
            if previous_transcription.lower().strip().endswith(transcription.lower().strip()):
                continue
            total_tokens = len(previous_transcription.split())
            elapsed_time = time() - tic
            tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
            previous_transcription += transcription
            shared_state["transcription"] = previous_transcription
            shared_state["tokens_per_sec"] = tokens_per_sec

        shared_state["stream"] = stream

# Gradio Interface
def create_interface():
    with gr.Blocks() as demo:
        audio_input = gr.Audio(source="microphone", type="numpy", streaming=True)
        model_input = gr.Textbox(value="model-name", label="Model")
        task_input = gr.Radio(choices=["transcribe", "translate"], value="transcribe", label="Task")
        temperature_input = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="Temperature")
        streaming_input = gr.Checkbox(label="Streaming Mode", value=True)

        transcription_output = gr.Textbox(label="Transcription")
        tps_output = gr.Textbox(label="Tokens per Second (TPS)")

        def update_transcription():
            return shared_state.get("transcription", ""), shared_state.get("tokens_per_sec", "")

        def start_task(audio, model, task, temperature, streaming):
            if audio:
                app.run(handle_audio(audio, task, streaming, temperature, model))
            return update_transcription()

        audio_input.change(start_task, [audio_input, model_input, task_input, temperature_input, streaming_input], [transcription_output, tps_output])

        gr.Interface(
            fn=start_task,
            inputs=[audio_input, model_input, task_input, temperature_input, streaming_input],
            outputs=[transcription_output, tps_output],
        ).launch()

if __name__ == "__main__":
    create_interface()