import io
import warnings

import numpy as np
from pydub import AudioSegment


def audio_from_bytes(audio_bytes, crop_min=0, crop_max=100):  # noqa: ANN201
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as e:
        msg = "Cannot load audio from bytes. Please ensure the audio data is in a supported format."
        raise RuntimeError(msg) from e
    if crop_min != 0 or crop_max != 100:
        audio_start = len(audio) * crop_min / 100
        audio_end = len(audio) * crop_max / 100
        audio = audio[audio_start:audio_end]
    data = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        data = data.reshape(-1, audio.channels)
    return audio.frame_rate, data

def audio_to_bytes(sample_rate, data, format="wav"):
    if format == "wav":
        data = convert_to_16_bit_wav(data)
    audio = AudioSegment(
        data.tobytes(),
        frame_rate=sample_rate,
        sample_width=data.dtype.itemsize,
        channels=(1 if len(data.shape) == 1 else data.shape[1]),
    )
    buffer = io.BytesIO()
    audio.export(buffer, format=format)
    return buffer.getvalue()

def convert_to_16_bit_wav(data):
    # This function remains unchanged
    warning = "Trying to convert audio automatically from {} to 16-bit int format."
    if data.dtype in [np.float64, np.float32, np.float16]:
        warnings.warn(warning.format(data.dtype))
        data = data / np.abs(data).max()
        data = data * 32767
        data = data.astype(np.int16)
    elif data.dtype == np.int32:
        warnings.warn(warning.format(data.dtype))
        data = data / 65536
        data = data.astype(np.int16)
    elif data.dtype == np.int16:
        pass
    elif data.dtype == np.uint16:
        warnings.warn(warning.format(data.dtype))
        data = data - 32768
        data = data.astype(np.int16)
    elif data.dtype == np.uint8:
        warnings.warn(warning.format(data.dtype))
        data = data * 257 - 32768
        data = data.astype(np.int16)
    elif data.dtype == np.int8:
        warnings.warn(warning.format(data.dtype))
        data = data * 256
        data = data.astype(np.int16)
    else:
        raise ValueError(
            "Audio data cannot be converted automatically from "
            f"{data.dtype} to 16-bit int format."
        )
    return data