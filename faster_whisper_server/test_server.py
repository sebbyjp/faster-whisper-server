import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
from real_time_transcription_server import RealTimeTranscriptionServer

class TestRealTimeTranscriptionServer(unittest.TestCase):

    def setUp(self):
        self.whisper_api_url = "http://test.whisper.api"
        self.agent1_url = "http://test.agent1.api"
        self.agent2_url = "http://test.agent2.api"
        self.server = RealTimeTranscriptionServer(
            self.whisper_api_url, self.agent1_url, self.agent2_url)

    @patch('sounddevice.InputStream')
    @patch('real_time_transcription_server.RealTimeTranscriptionServer.process_audio_chunk')
    def test_handle_audio_stream(self, mock_process_audio_chunk, mock_input_stream):
        # Mock the InputStream context manager
        mock_input_stream.return_value.__enter__.return_value = MagicMock()

        # Set up an event loop and run the handle_audio_stream coroutine
        asyncio.run(self.server.handle_audio_stream())

        # Ensure the InputStream was started
        mock_input_stream.assert_called_once()

        # Verify that audio_callback was scheduled to process audio chunks
        self.assertTrue(mock_process_audio_chunk.called)

    @patch('real_time_transcription_server.aiohttp.ClientSession')
    def test_transcribe_audio(self, mock_client_session):
        # Mock the session and response from Whisper
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_client_session.return_value.__aenter__.return_value = mock_session
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.content = AsyncMock()
        mock_response.content.__aiter__.return_value = [
            '{"transcription": "hello"}'.encode(),
            '{"transcription": "world"}'.encode()
        ]

        # Run the transcribe_audio coroutine
        transcriptions = []
        async def collect_transcriptions():
            async for word in self.server.transcribe_audio(b'audio_chunk'):
                transcriptions.append(word)
        asyncio.run(collect_transcriptions())

        # Check if the transcription results are as expected
        self.assertEqual(transcriptions, ["hello", "world"])

    @patch('real_time_transcription_server.aiohttp.ClientSession')
    def test_process_agent(self, mock_client_session):
        # Mock the session and response from the agent
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_client_session.return_value.__aenter__.return_value = mock_session
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.content = AsyncMock()
        mock_response.content.__aiter__.return_value = [
            '{"response": "agent response 1"}'.encode(),
            '{"response": "agent response 2"}'.encode()
        ]

        # Run the process_agent coroutine
        agent_responses = []
        async def collect_responses():
            async for response in self.server.process_agent(self.agent1_url, "test instruction"):
                agent_responses.append(response)
        asyncio.run(collect_responses())

        # Check if the agent responses are as expected
        self.assertEqual(agent_responses, ["{\"response\": \"agent response 1\"}", "{\"response\": \"agent response 2\"}"])

    @patch('real_time_transcription_server.pyttsx3.Engine')
    def test_stream_speech(self, mock_engine):
        # Mock the text-to-speech engine
        mock_engine.return_value = MagicMock()

        # Sample agent stream data
        async def mock_agent_stream():
            yield "This is a test response."

        # Run the stream_speech coroutine
        asyncio.run(self.server.stream_speech(mock_agent_stream()))

        # Ensure the speech engine spoke the correct chunks
        engine_instance = mock_engine.return_value
        engine_instance.say.assert_called_with("This is a")
        self.assertEqual(engine_instance.say.call_count, 1)
        engine_instance.runAndWait.assert_called_once()

    @patch('real_time_transcription_server.RealTimeTranscriptionServer.extract_json')
    def test_stream_json(self, mock_extract_json):
        # Sample agent stream data
        async def mock_agent_stream():
            yield '{"key": "value"}'

        # Mock the JSON extraction function
        mock_extract_json.return_value = {"key": "value"}

        # Run the stream_json coroutine
        asyncio.run(self.server.stream_json(mock_agent_stream()))

        # Ensure JSON was extracted correctly
        mock_extract_json.assert_called_once_with('{"key": "value"}')

if __name__ == "__main__":
    unittest.main()