from collections.abc import Callable
from functools import partial
import json
import logging
import os
from pathlib import Path
import sys
from typing import ClassVar, Literal, Union

import click
from pydantic import AnyHttpUrl, Field, FilePath
from pydub import AudioSegment
from pytube import YouTube
import requests
from TTS.api import TTS
import yt_dlp

from faster_whisper_server.agents.speaker_config import SpeakerConfig as TTSCConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def json_to_netscape_string(json_content: str) -> str:
    """Convert JSON-formatted cookies to Netscape format string."""
    json_content = Path(json_content).read_text() if Path(json_content).exists() else json_content
    cookies = json.loads(json_content)
    netscape_cookies = "# Netscape HTTP Cookie File\n\n"

    for cookie in cookies:
        domain = cookie.get("domain", "")
        flag = "TRUE" if cookie.get("hostOnly", False) is False else "FALSE"
        path = cookie.get("path", "/")
        secure = "TRUE" if cookie.get("secure", False) else "FALSE"
        expiration = str(cookie.get("expirationDate", "0")).split(".")[0]
        name = cookie.get("name", "")
        value = cookie.get("value", "")

        netscape_cookies += f"{domain}\t{flag}\t{path}\t{secure}\t{expiration}\t{name}\t{value}\n"
    Path("cookies.txt").write_text(netscape_cookies)
    return "cookies.txt"




def load_audio_wav_or_url(audio: str, cookies: str | None = None) -> str:
    """Load audio from a file path or URL, converts file or URL (like YouTube) to a WAV file."""

    def download_audio(url: str, output_path: str) -> str:
        """Download audio from a given URL."""
        response = requests.get(url, timeout=10)
        with Path(output_path).open("wb") as f:
            f.write(response.content)
            if not str(output_path).endswith(".wav"):
                output_path = convert_to_wav(output_path, output_path.replace(".mp3", ".wav"))
        return output_path

    def convert_to_wav(input_path: str, output_path: str) -> str:
        """Convert any audio file to WAV format."""
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return output_path

    def download_youtube_audio(url: str, output_path: str) -> str:
        """Download audio from a YouTube URL using yt-dlp."""
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_path + ".%(ext)s",  # Save output as a .wav file
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",  # Use WAV for lossless audio
                    "preferredquality": "0",  # '0' means best quality available
                }
            ],
        }

        if cookies:
            ydl_opts["cookiefile"] = json_to_netscape_string(cookies)
        logging.debug("Downloading audio from YouTube: %s for cookies %s", url, cookies)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path + ".wav"

    # Determine if the input is a YouTube URL or a regular HTTP URL
    if "youtube.com" in str(audio) or "youtu.be" in str(audio):
        # Download the audio from the YouTube URL
        temp_audio_path = "temp_audio.wav"
        audio_path = download_youtube_audio(str(audio), temp_audio_path)
    elif isinstance(audio, str) and audio.startswith("http"):
        # Download the audio from a regular URL
        temp_audio_path = "temp_audio"
        audio_path = download_audio(str(audio), temp_audio_path)
    else:
        # Use the provided file path if it's local
        audio_path = audio
    if not str(audio_path).endswith(".wav"):
        audio_path = convert_to_wav(audio_path, audio_path.replace(Path(audio_path).suffix, ".wav"))
    return audio_path


def setup_voice_cloning(
    reference_audio: FilePath | AnyHttpUrl,
    model: str | None = "TTSCConfig.DEFAULT_VOICE_CLONING_MODEL",
    vocoder_model: str | None = None,
    gpu: bool = True,
    cookies: str | None = None,
    config_path: str | None = None,
) -> TTS:
    vocoder_model = "vocoder_models/universal/libri-tts/wavegrad" if vocoder_model == "default" else vocoder_model
    reference_audio = load_audio_wav_or_url(reference_audio, cookies=cookies)
    tts = TTS(gpu=gpu, config_path=config_path) if config_path else TTS(model, gpu=gpu)
    tts.vc = partial(tts.tts, speaker_wav=reference_audio)
    tts.reference_audio = reference_audio
    return tts



@click.command()
@click.argument("text", type=str, default="")
@click.option(
    "--reference_audio",
    "-r",
    type=str,
    help="Reference audio file path or URL",
    default=None,
)
@click.option("--model", type=str, help="TTS model path", default=None)
@click.option("--language", "-l", type=str, help="Language code", default=None)
@click.option("--cookies", "-c", type=str, help="Path to cookies file", default=None)
@click.option("--debug", "-d", is_flag=True, help="Enable debug logging")
@click.option("--config_path", "-cp", type=str, help="Path to a JSON configuration file", default=None)
def cli(
    text: str = None,
    reference_audio: FilePath | AnyHttpUrl | None = None,
    model: str | None = None,
    language: str | None = None,
    cookies: str | None = None,
    debug: bool = False,
    config_path: str | None = None,
):
    if sys.flags.debug or debug:
        logging.basicConfig(level=logging.DEBUG)
    if not config_path and Path("tts_config.json").exists():
        config_path = "tts_config.json"
        click.echo("Using config file: {config_path}")
    else:
        click.echo("No config file found, using default settings.")
    tts_config = TTSCConfig(
        model=model or TTSCConfig.DEFAULT_VOICE_CLONING_MODEL,
        # first_speaker=reference_audio or TTSCConfig.DEFAULT_REFERENCE_AUDIO,
        first_language=language or "en",
        # config_path=config_path,
    )
    text = text or """Here is an exerp from the book 'The Great Gatsby' by F Scott Fitzgerald. 
    In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since.
    Whenever you feel like criticizing any one, he told me, just remember that all the people in this world haven't had the advantages that you've had. 
    He didn't say any more but we've always been unusually communicative in a reserved way, and I understood that he meant a great deal more than that.
    In consequence I'm inclined to reserve all judgments, a habit that has opened up many curious natures to me and also made me the victim of not a few veteran bores.
    The abnormal mind is quick to detect and attach itself to this quality when it appears in a normal person, and so it came about that in college I was unjustly accused of being a politician.
    """
    # vc = setup_voice_cloning(tts_config.first_speaker, tts_config.model, gpu=tts_config.gpu, cookies=cookies)
    tts =  setup_voice_cloning( tts_config.model, gpu=tts_config.gpu)

    if hasattr(tts, "is_multilingual"):
        text = tts.tts_to_file(text, language=tts_config.first_language, split_sentences=True)
    else:
        text = tts.tts_to_file(text, language=tts_config.first_language, split_sentences=False)
        # text = vc.tts_to_file(text,  language=tts_config.first_language, speaker_wav=vc.reference_audio)
    click.echo("Audio saved to output.wav")


if __name__ == "__main__":
    cli()
