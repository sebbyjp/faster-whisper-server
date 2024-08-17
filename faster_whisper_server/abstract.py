from abc import ABC, abstractmethod
import asyncio
import json

import aiohttp
import pyttsx3
import sounddevice as sd


class AbstractRealTimeTranscriptionServer(ABC):
    def __init__(self, whisper_api_url, agent1_url, agent2_url, samplerate=16000, blocksize=1024):
        self.whisper_api_url = whisper_api_url
        self.agent1_url = agent1_url
        self.agent2_url = agent2_url
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.current_sentence = ""

    @abstractmethod
    async def handle_audio_stream(self):
        """Start capturing audio and processing it."""
        pass

    @abstractmethod
    def audio_callback(self, indata, frames, time, status):
        """Callback function for processing audio data."""
        pass

    @abstractmethod
    async def process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio data."""
        pass

    @abstractmethod
    async def transcribe_audio(self, audio_chunk):
        """Send audio data to the transcription service and yield the transcribed words."""
        pass

    @abstractmethod
    def is_instruction(self, sentence):
        """Evaluate if the current sentence is a complete instruction."""
        pass

    @abstractmethod
    async def process_agent(self, agent_url, instruction):
        """Send an instruction to an agent and yield the response."""
        pass

    @abstractmethod
    async def stream_speech(self, agent_stream):
        """Stream the agent's response as speech."""
        pass

    @abstractmethod
    async def stream_json(self, agent_stream):
        """Stream the agent's response as JSON."""
        pass

    @abstractmethod
    def extract_chunk(self, chunk):
        """Extract the text chunk that is ready to be spoken."""
        pass

    @abstractmethod
    def extract_json(self, chunk):
        """Extract and return JSON from the streamed data."""
        pass

    @abstractmethod
    async def start(self):
        """Start the server and begin processing."""
        pass



class RealTimeTranscriptionServer(AbstractRealTimeTranscriptionServer):
    def __init__(self, whisper_api_url, agent1_url, agent2_url, samplerate=16000, blocksize=1024):
        super().__init__(whisper_api_url, agent1_url, agent2_url, samplerate, blocksize)
        self.speech_engine = pyttsx3.init()

    async def handle_audio_stream(self):
        with sd.InputStream(callback=self.audio_callback, samplerate=self.samplerate, blocksize=self.blocksize):
            await asyncio.sleep(60 * 60)  # Keep the stream open for 1 hour

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, flush=True)
        audio_chunk = indata.tobytes()
        asyncio.create_task(self.process_audio_chunk(audio_chunk))

    async def process_audio_chunk(self, audio_chunk):
        async for transcribed_word in self.transcribe_audio(audio_chunk):
            self.current_sentence += f" {transcribed_word}".strip()
            if self.is_instruction(self.current_sentence):
                instruction = self.current_sentence
                self.current_sentence = ""
                agent1_stream = self.process_agent(self.agent1_url, instruction)
                agent2_stream = self.process_agent(self.agent2_url, instruction)
                asyncio.create_task(self.stream_speech(agent1_stream))
                asyncio.create_task(self.stream_json(agent2_stream))

    async def transcribe_audio(self, audio_chunk):
        async with aiohttp.ClientSession() as session:  # noqa: SIM117
            async with session.post(self.whisper_api_url, data=audio_chunk) as response:
                async for line in response.content:
                    transcribed_word = json.loads(line.decode())["transcription"]
                    yield transcribed_word

    def is_instruction(self, sentence):
        return sentence.endswith(".")

    async def process_agent(self, agent_url, instruction):
        async with aiohttp.ClientSession() as session:
            async with session.post(agent_url, json={"instruction": instruction}) as response:
                async for line in response.content:
                    yield line.decode()

    async def stream_speech(self, agent_stream):
        async for chunk in agent_stream:
            chunk_text = self.extract_chunk(chunk)
            self.speech_engine.say(chunk_text)
            self.speech_engine.runAndWait()

    async def stream_json(self, agent_stream):
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

    async def start(self):
        await self.handle_audio_stream()

# Run the concrete server
if __name__ == "__main__":
    whisper_api_url = "http://localhost:8000/whisper"
    agent1_url = "http://localhost:8001/agent1"
    agent2_url = "http://localhost:8002/agent2"

    server = RealTimeTranscriptionServer(whisper_api_url, agent1_url, agent2_url)
    asyncio.run(server.start())