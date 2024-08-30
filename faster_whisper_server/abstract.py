from abc import ABC, abstractmethod
import asyncio
import json

import pyttsx3


class AbstractRealTimeTranscriptionServer(ABC):
    def __init__(self, whisper_api_url, agent1_url, agent2_url, samplerate=16000, blocksize=1024) -> None:
        self.whisper_api_url = whisper_api_url
        self.agent1_url = agent1_url
        self.agent2_url = agent2_url
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.current_sentence = ""

    @abstractmethod
    async def handle_audio_stream(self):
        """Start capturing audio and processing it."""

    @abstractmethod
    def audio_callback(self, indata, frames, time, status):
        """Callback function for processing audio data."""

    @abstractmethod
    async def process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio data."""

    @abstractmethod
    async def transcribe_audio(self, audio_chunk):
        """Send audio data to the transcription service and yield the transcribed words."""

    @abstractmethod
    def is_instruction(self, sentence):
        """Evaluate if the current sentence is a complete instruction."""

    @abstractmethod
    async def process_agent(self, agent_url, instruction):
        """Send an instruction to an agent and yield the response."""

    @abstractmethod
    async def stream_speech(self, agent_stream):
        """Stream the agent's response as speech."""

    @abstractmethod
    async def stream_json(self, agent_stream):
        """Stream the agent's response as JSON."""

    @abstractmethod
    def extract_chunk(self, chunk):
        """Extract the text chunk that is ready to be spoken."""

    @abstractmethod
    def extract_json(self, chunk):
        """Extract and return JSON from the streamed data."""

    @abstractmethod
    async def start(self):
        """Start the server and begin processing."""



class RealTimeTranscriptionServer(AbstractRealTimeTranscriptionServer):
    def __init__(self, whisper_api_url, agent1_url, agent2_url, samplerate=16000, blocksize=1024) -> None:
        super().__init__(whisper_api_url, agent1_url, agent2_url, samplerate, blocksize)
        self.speech_engine = pyttsx3.init()



    def is_instruction(self, sentence):
        return sentence.endswith(".")


    async def stream_speech(self, agent_stream) -> None:
        async for chunk in agent_stream:
            chunk_text = self.extract_chunk(chunk)
            self.speech_engine.say(chunk_text)
            self.speech_engine.runAndWait()

    async def stream_json(self, agent_stream) -> None:
        async for chunk in agent_stream:
            json_content = self.extract_json(chunk)
            print(f"Received JSON content: {json_content}")

    def extract_chunk(self, chunk):
        return " ".join(chunk.split()[:3])

    def extract_json(self, chunk):
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            print("Incomplete JSON chunk, waiting for more data.")
            return {}

    async def start(self) -> None:
        await self.handle_audio_stream()

# Run the concrete server
if __name__ == "__main__":
    whisper_api_url = "http://localhost:8000/whisper"
    agent1_url = "http://localhost:8001/agent1"
    agent2_url = "http://localhost:8002/agent2"

    server = RealTimeTranscriptionServer(whisper_api_url, agent1_url, agent2_url)
    asyncio.run(server.start())
