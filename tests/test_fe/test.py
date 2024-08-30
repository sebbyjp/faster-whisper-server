import json
import logging
from pprint import pprint
import time

from mbodied.agents import LanguageAgent
from openai import OpenAI
import pytest

SYSTEM_PROMPT = """
You are a brand new assistant...
"""

DETERMINE_INSTRUCTION_PROMPT = """
Determine if the following is an instruction or direct question. Answer yes if it is either, and no if it is neither.
"""

NOT_A_COMPLETE_INSTRUCTION = "Not a complete instruction."


@pytest.fixture
def query() -> str:
    return "Tell me about the weather in New York. Now."


def test_determine_instruction(query) -> None:
    weak_agent = LanguageAgent(
        model_src="openai", api_key="mbodi-demo-1",
        model_kwargs={"base_url": "https://api.mbodi.ai/v1"},
        context=SYSTEM_PROMPT
    )
    response = weak_agent.act(DETERMINE_INSTRUCTION_PROMPT + query, model="astroworld",
                                extra_body={"guided_choice": ["Yes", "No"]})
    assert response == "Yes", f"Expected 'Yes', got {response}"
    print("Test passed.")
    response = weak_agent.act("Why was it a complete instruction?", model="astroworld")
    print(response)


def byte_stream_generator(response):
    """Generator function that yields a stream of bytes from the API response."""
    for byte_chunk in response.iter_bytes(chunk_size=4096):
        if byte_chunk:
            yield byte_chunk
        else:
            logging.warning("Skipped an empty or corrupted packet")


def stream_tts(text_chunks, openai_api: OpenAI, out_file="out.wav") -> None:
    start_time = time.time()
    with open(out_file, "wb") as out:
      for text in text_chunks:
        print(f"Generating audio for: {text}")
        with openai_api.audio.speech.with_streaming_response.create(
            model="tts-1-hd",
            voice="alloy",
            response_format="wav",  # similar to WAV, but without a header chunk at the start.
            input=text,
        ) as response:
            print(f"Time to first byte: {int((time.time() - start_time) * 1000)}ms")
            for chunk in response.iter_bytes(chunk_size=1024):
              out.write(chunk)

def test_act_and_stream() -> None:
    text = "Tell me about the weather in New York. Now."
    agent = LanguageAgent(model_src="openai", api_key="mbodi-demo-1", model_kwargs={"base_url": "https://api.mbodi.ai/v1"})
    response = ""
    for r in agent.act_and_stream(text, model="astroworld"):
        response += r
        print(response)

def test_speech(query) -> None:
    openai_api = OpenAI(  api_key="mbodi-demo-1", base_url="http://localhost:3389")
    # split query into chunks of three words
    text_chunks = [" ".join(query.split()[i:i+3]) for i in range(0, len(query.split()), 3)]
    stream_tts(text_chunks, openai_api)


def test_guided_json():
    weak_agent = LanguageAgent(
        model_src="openai", api_key="mbodi-demo-1",
        model_kwargs={"base_url": "https://api.mbodi.ai/v1",
                        "response_format": {
                            "type": "json_object",
                        },

        },
        context="You are a brand new assistant... that responds with a json schema.",
    )

    from mbodied.types.motion.control import MobileSingleArmControl
    # from mbodied.types.motion.control import HandControl

    # response = weak_agent.act("Look up and move hand left." + query, model="astroworld",
    #                             extra_body={"guided_json": MobileSingleArmControl().schema()})

    # print(response)
    pprint(MobileSingleArmControl().schema(include_descriptions=True))
    # assert json.loads(response) is not None, f"Expected a json object with a key, got {response}"
    # pprint(f"Test passed. Response: {response}")

if __name__ == "__main__":
    query = "Tell me about the weather in New York. What is the weather like in New York?"
    # test_determine_instruction(query)
    test_guided_json()
    # test_act_and_stream()
