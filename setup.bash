#!/bin/bash
# Variables
MODEL="Systran/faster-whisper-large-v3"
LANGUAGE="en"
RESPONSE_FORMAT="json"
TEMPERATURE="0"
FILE_PATH="/home/ubuntu/seb/audio/faster-whisper-server/faster_whisper_server/output.wav"  # Replace with the actual file path
PCM_FILE_PATH="./audio.pcm"  # Replace with the desired PCM file path
TRANSCRIBE_ENDPOINT="http://0.0.0.0:7543/v1/audio/transcriptions"
TRANSLATE_ENDPOINT="http://0.0.0.0:7543/v1/audio/translations"

# Function to install bc if not installed
install_bc() {
    if ! command -v bc &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y bc
    else
        echo "bc is already installed"
    fi
}

# Function to install ffmpeg version 7 or above
install_ffmpeg() {
    if command -v ffmpeg &> /dev/null; then
        ffmpeg_version=$(ffmpeg -version | grep "ffmpeg version" | awk '{print $3}' | cut -d'-' -f1)
        
        if [ -n "$ffmpeg_version" ]; then
            # Check if ffmpeg_version starts with 7 or is greater than 7
            if [[ "$ffmpeg_version" == 7* || "$ffmpeg_version" > 7 ]]; then
                echo "FFmpeg version $ffmpeg_version is already installed"
                return
            fi
        fi
    fi

    echo "Installing FFmpeg version 7 or above..."
    sudo add-apt-repository ppa:savoury1/ffmpeg4 -y
    sudo add-apt-repository ppa:savoury1/ffmpeg5 -y
    sudo add-apt-repository ppa:savoury1/ffmpeg6 -y
    sudo add-apt-repository ppa:savoury1/ffmpeg7 -y
    sudo apt-get update
    sudo apt-get install ffmpeg -y
}

# Function to convert audio file to PCM format
convert_to_pcm() {
    ffmpeg -i "$FILE_PATH" -f s16le -acodec pcm_s16le -ar 16000 -ac 1 "$PCM_FILE_PATH"
}

# Function to stream the PCM file to the transcription endpoint
stream_transcribe_file() {
    cat "$PCM_FILE_PATH" | curl -X POST "$TRANSCRIBE_ENDPOINT" \
        -H "Content-Type: application/octet-stream" \
        -H "model: $MODEL" \
        -H "language: $LANGUAGE" \
        -H "response_format: $RESPONSE_FORMAT" \
        -H "temperature: $TEMPERATURE" \
        --data-binary @-
}

# Function to stream the PCM file to the translation endpoint
stream_translate_file() {
    cat "$PCM_FILE_PATH" | curl -X POST "$TRANSLATE_ENDPOINT" \
        -H "Content-Type: application/octet-stream" \
        -H "model: $MODEL" \
        -H "response_format: $RESPONSE_FORMAT" \
        -H "temperature: $TEMPERATURE" \
        --data-binary @-
}

# Main script execution
install_bc
install_ffmpeg
convert_to_pcm
stream_transcribe_file
stream_translate_file