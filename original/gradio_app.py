import io
import os
import logging
from collections.abc import Generator, AsyncGenerator
from tempfile import NamedTemporaryFile
from typing import Tuple
import wave
import gradio as gr
import numpy as np
from openai import OpenAI
import asyncio
import json
import websockets
import websockets.client
import websockets.connection
from websockets.client import WebSocketClientProtocol
import soundfile as sf
from faster_whisper_server.config import Config, Task


logger = logging.getLogger(__name__)



CUSTOM_CSS = """
#footer {display: none !important;}
"""
THEME = gr.themes.Soft(
    primary_hue="red",
    secondary_hue="stone",
    neutral_hue="zinc",
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

async def send_audio_stream(websocket: WebSocketClientProtocol, audio_data, sample_rate):
    chunk_size = 1024
    print(f"Sending audio data with shape {audio_data.shape} and sample rate {sample_rate}")
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        await websocket.send(chunk.tobytes())
        with wave.open("sent_audio2.wav", 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(chunk.tobytes())

def create_gradio_demo(config: Config) -> gr.Blocks:
    ws_host = os.getenv("AUDIO_HOST", "api.mbodi.ai/audio")
    port = int(os.getenv("AUDIO_PORT", "7543"))
    # openai_client = OpenAI(base_url=f"{http_host}:{port}/v1", api_key="mbodi-demo")


    async def handler(stream, audio: Tuple[int, np.ndarray], model: str, task: str, temperature: float) -> AsyncGenerator[str, None]:
        if not audio:
            yield "", None
        

        sr, y = audio
        y_mono: np.ndarray = np.mean(y, axis=1) if len(y.shape) > 1 and y.shape[1] > 1 else y
        y_mono = y_mono.astype(np.float32)
        y_mono /= np.max(np.abs(y_mono))
        stream = np.concatenate([stream, y_mono])
        uri = f"ws://{ws_host}:{port}/v1/transcriptions?model={model}&task={task}&temperature={temperature}"
        wav_io = io.BytesIO()
        wav_io.name = "audio_file.wav"
        sf.write(wav_io, y_mono, sr, format='WAV')
        wav_io.seek(0)  # Reset pointer to the beginning

        async with websockets.connect(uri) as websocket:
            print(f"Sending audio data to {uri}")
            send_task = asyncio.create_task(send_audio_stream(websocket, y_mono, sr))
            full_transcription = ""
            try:
                async for message in websocket:
                    print(f"Received message: {message}")
                    transcription = json.loads(message)
                    if 'text' in transcription:
                        full_transcription += transcription['text'] + " "
                        yield  full_transcription.strip()
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                send_task.cancel()
        yield  full_transcription.strip()
        # audio_file_wav = "audio_file.wav"
        # audio_file_mp3 = "audio_file.mp3"

        # # Write to WAV file using soundfile
        # sf.write(audio_file_wav, stream, sr)
        # async with websockets.connect(uri) as websocket:
        #     # Start sending audio data
        #     print(f"Sending audio data to {uri}")
        #     send_task = asyncio.create_task(send_audio_stream(websocket, y_mono, sr))
            
        #     # Receive and process transcriptions
        #     full_transcription = ""
        #     try:
        #         async for message in websocket:
        #             print(f"Received message: {message}")
        #             transcription = json.loads(message)
        #             if 'text' in transcription:
        #                 full_transcription += transcription['text'] + " "
        #                 yield  full_transcription.strip()
        #     except websockets.exceptions.ConnectionClosed:
        #         pass
        #     finally:
        #         send_task.cancel()
        # yield  full_transcription.strip()
    def update_model_dropdown() -> gr.Dropdown:
        try:
            models = openai_client.models.list().data
            model_names: list[str] = [model.id for model in models]
            if config.whisper.model not in model_names:
                model_names.append(config.whisper.model)
            recommended_models = {model for model in model_names if model.startswith("Systran")}
            other_models = [model for model in model_names if model not in recommended_models]
            model_names = list(recommended_models) + other_models
        except Exception as e:
            logger.error(f"Error fetching models: {str(e)}")
            model_names = [config.whisper.model]
        
        return gr.Dropdown(
            choices=model_names,
            label="Model",
            value=config.whisper.model,
        )

    model_dropdown = gr.Dropdown(
        choices=[config.whisper.model],
        label="Model",
        value=config.whisper.model,
    )
    task_dropdown = gr.Dropdown(
        choices=[task.value for task in Task],
        label="Task",
        value=Task.TRANSCRIBE,
    )
    temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.0)

    with gr.Blocks(title="Whisper Streaming", theme=THEME, css=CUSTOM_CSS) as demo:
        gr.Markdown("# Real Time Audio Transcription")
        with gr.Row():
            with gr.Column(scale=2):
                audio_in = gr.Audio(label="Audio Input", streaming=True, sources=["microphone"], type="numpy", interactive=True, show_download_button=True)
                model_dropdown = gr.Dropdown(
                    choices=[config.whisper.model],
                    label="Model",
                    value=config.whisper.model,
                )
                task_dropdown = gr.Dropdown(
                    choices=[task.value for task in Task],
                    label="Task",
                    value=Task.TRANSCRIBE,
                )
                temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.0)
            with gr.Column(scale=3):
                output_text = gr.Textbox(label="Transcription", lines=10)
        
        gr.on([model_dropdown.change, task_dropdown.change], fn=update_model_dropdown) 
        audio_in.stream(handler, inputs=[audio_in,model_dropdown,model_dropdown, temperature_slider],  outputs=[output_text])


    return demo

config = Config()
with  create_gradio_demo(config) as demo:
    demo.launch(server_name="0.0.0.0",server_port=7542, share=False, ssl_verify=False)

