import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.generation import SUPPORTED_LANGS
import numpy as np

loaded = False
# Preload models (you might want to call this separately if it takes too long)
def preload_tts():
    global loaded
    if not loaded:
        preload_models(text_use_gpu=True, fine_use_gpu=True, coarse_use_gpu=True, codec_use_gpu=True)
        loaded = True


# Set up voice options
PROMPT_LOOKUP = {}
for _, lang in SUPPORTED_LANGS:
    for n in range(10):
        label = f"Speaker {n} ({lang})"
        PROMPT_LOOKUP[label] = f"{lang}_speaker_{n}"
PROMPT_LOOKUP["Unconditional"] = None
PROMPT_LOOKUP["Announcer"] = "announcer"


def generate_tts(text, voice_preset="Speaker 1 (en)", text_temp=0.7, waveform_temp=0.7):
    """Generate text-to-speech audio using the Bark model.

    :param text: The input text to convert to speech
    :param voice_preset: The voice preset to use (e.g., "Speaker 9 (en)", "Announcer")
    :param text_temp: Text temperature for generation
    :param waveform_temp: Waveform temperature for generation
    :return: A tuple containing the sample rate and the audio array
    """
    history_prompt = PROMPT_LOOKUP.get(voice_preset, voice_preset)

    audio_array = generate_audio(text, history_prompt=history_prompt,
                                 text_temp=text_temp, waveform_temp=waveform_temp)

    # Convert to 16-bit int values
    audio_array = (audio_array * 32767).astype(np.int16)

    return SAMPLE_RATE, audio_array

# Example usage
if __name__ == "__main__":
    text = "No way actually?."
    sample_rate, audio = generate_tts(text)
    # print(f"Generated audio with sample rate {sample_rate} Hz")
    # print(f"Audio array shape: {audio.shape}")
    
    # To save the audio, you can use a library like scipy or soundfile
    # For example, using scipy:
    from scipy.io import wavfile
    wavfile.write("output.wav", sample_rate, audio)