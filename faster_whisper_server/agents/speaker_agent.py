import re
from time import time
from typing import ClassVar, Iterator, Literal

import gradio as gr
from gradio import Audio, Text
import numpy as np
from pydantic import AnyHttpUrl, Field, FilePath
from rich.console import Console
import soundfile as sf
from TTS.api import TTS

from faster_whisper_server.agents.agent import StatefulAgent
from faster_whisper_server.agents.config import AgentConfig, CompletionConfig, State, persist_maybe_clear

Language =Literal["en", "es", "fr", "de", "it", "ja", "ko", "nl", "pl", "pt", "ru", "zh"]
Speaker = Literal[
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

def speaker_pre_process(prompt: str, local_state: State, shared_state: State | None = None) -> str:
    local_state.clear()
    text = prompt
    if text and len(text.split()) < 2 and not (text.endswith((".", "?", "!"))) and shared_state["act_mode"] != "repeat":
        return
    if shared_state.get("speaker_status") == "wait":
        return
    return text


def speaker_post_process(prompt: str, response: str, local_state: State, shared_state: State | None = None) -> str:
    """Postprocess the data before returning it."""
    shared_state.clear()
    shared_state["speaker_status"] = "done"
    shared_state["actor_status"] = "wait"
    shared_state["instruct_status"] = "ready"
    shared_state["whisper_status"] = "ready"

    shared_state.update(local_state)
    return np.ndarray([0], dtype=np.int16)


def export_onnx(model: str, output: str) -> None:
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import Vits

    config = VitsConfig()
    config.load_json("config.json")

    # Initialize VITS model and load its checkpoint
    vits = Vits.init_from_config(config)
    vits.load_checkpoint(config, "]best_model.pth")
    vits.export_onnx(output_path='model.onnx')



def setup_tts(
    model: str | None = "tts_models/multilingual/multi-dataset/xtts_v2",

    gpu: bool = True,
    config_path: str | None = None,
) -> TTS:
    # vocoder_model = "vocoder_models/universal/libri-tts/wavegrad" if vocoder_model == "default" else vocoder_model
    return TTS( gpu=gpu, config_path=config_path) if config_path else TTS(model, gpu=gpu) 

if gr.NO_RELOAD:
    tts = setup_tts()

class SpeakerConfig(AgentConfig):
    DEFAULT_MODEL: ClassVar[str] = "tts_models/multilingual/multi-dataset/xtts_v2"
    DEFAULT_VOICE_CLONING_MODEL: ClassVar[str] = "tts_models/multilingual/multi-dataset/your_tts"
    DEFAULT_REFERENCE_AUDIO: ClassVar[AnyHttpUrl] = "https://www.youtube.com/watch?v=Ij0ZmgG6wCA"
    model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    first_speaker: Speaker = "Luis Moray"

    # second_speaker: AnyHttpUrl = "https://www.youtube.com/watch?v=Ij0ZmgG6wCA"
    chunk_length_ms: int = 500
    first_language: Language = "en"
    # second_language: Languages = "en"
    config_path: str | None = None
    gpu: bool = True
    completion_config: CompletionConfig = CompletionConfig(
        pre_process=speaker_pre_process,
        post_process=speaker_post_process,
    )
speaker_config = SpeakerConfig(
    DEFAULT_VOICE_CLONING_MODEL="tts_models/multilingual/multi-dataset/your_tts",
    DEFAULT_REFERENCE_AUDIO="https://www.youtube.com/watch?v=Ij0ZmgG6wCA",
    model="tts_models/multilingual/multi-dataset/xtts_v2",
    first_speaker="Luis Moray",
    # second_speaker="https://www.youtube.com/watch?v=Ij0ZmgG6wCA",
    chunk_length_ms=500,
    first_language="en",
    # second_language="en",
    # config_path="tts_config.json",
    gpu=True,
    completion_config=CompletionConfig(
        pre_process=speaker_pre_process,
        post_process=speaker_post_process,
    ),
)

console = Console(style="bold white on blue")

class SpeakerAgent(StatefulAgent):
    config: SpeakerConfig
    def handle_stream(self, text: str | None, config: AgentConfig, local_state: State, shared_state: State ) -> Iterator[tuple[bytes, str, dict]]:  # noqa: E501
        """Generate and stream TTS audio using Coqui TTS with diarization."""
        console.print(f"speak STATE: {local_state}")
        if not text:
            return None

        sentences = [sentence.strip() + punct for sentence, punct in re.findall(r"([^.!?]*)([.!?])", text) if sentence.strip()]
        console.print(f"Sentences: {sentences}, spoken_idx:, {local_state.spoken_idx}")


        speaker, language = config.first_speaker, config.first_language
        sr = tts.synthesizer.output_sample_rate

        for idx, sentence in enumerate(sentences):
            if sentence and sentence not in local_state.spoken:
                audio_array = (np.array(tts.tts(sentence, speaker=speaker, language=language, split_sentences=False)) * 32767).astype(np.int16)
                # Log the sentence and mark the state as speaking
                console.print(f"SPEAK Text: {sentence}", style="bold white on blue")
                local_state.spoken += sentence
                audio_seconds = len(audio_array) / float(sr)
                console.print(f"Audio seconds: {audio_seconds}", style="bold white on blue")
                local_state.update({
                    "audio_array": audio_array,
                    "spoken_idx": idx + 1,  # Move to the next sentence
                    "audio_finish_time": time() + audio_seconds,
                })
                shared_state.update(audio_finish_time=local_state.audio_finish_time)
                shared_state.speaker_status = "speaking"
                f = f"out{idx}.wav"
                sf.write(file=f, data=audio_array, samplerate=sr)
                return f
        return local_state.spoken



