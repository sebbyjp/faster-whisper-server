import os

from pydantic import Field
from pydantic.json_schema import JsonSchemaValue
from pydantic.types import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Callable, Dict, TypedDict, dataclass_transform  # noqa: UP035

@dataclass_transform()
class State(TypedDict, total=False):
    pass

@dataclass_transform()
class Guidance(TypedDict, total=False):
    choices: list[str] | None = Field(default=None, examples=[lambda: ["Yes", "No"]])
    json: str | Dict| JsonSchemaValue | None = Field(
      default=None,
      description="The json schema as a dict or string.",
      examples=[{"type": "object", "properties": {"key": {"type": "string"}}}],
    )


class BaseAgentConfig(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=False)
    base_url: str = "https://api.mbodi.ai/v1"
    auth_token: SecretStr = os.getenv("MBODI_API_KEY", "mbodi-demo-1")


@dataclass_transform()
class CompletionConfig(TypedDict, total=False):
    guidance: Guidance | None = Field(default=None, examples=[Guidance(choices=["Yes", "No"])])
    pre_process: Callable[[str, State], str] | BaseAgentConfig | None = Field(default=None, description="A callable or agent that takes the prompt and state and returns a modified prompt.")
    post_process: Callable[[str, str, State], str] | BaseAgentConfig| None = Field(default=None, examples=[lambda prompt, response: prompt if response == "yes" else ""],
        description="A callable or agent that takes the prompt, response, and state and returns a modified prompt.")

    prompt: str | None | Callable = Field(
      default="Give a command for how the robot should move in the following json format:",
      examples=[lambda x, y: f"Translate the following text to {x} if it is {y} and vice versa. Respond with only the translated text and nothing else."],
      description="The prompt to be used in the completion. Use a callable if you want to add extra information at the time of agent action."
    )
    reminder: str | None = Field(default=None, examples=["Remember to respond with only the translated text and nothing else."])



class AgentConfig(BaseAgentConfig):
    base_url: str = "https://api.mbodi.ai/v1"
    auth_token: str = "mbodi-demo-1"
    completion: CompletionConfig = Field(default_factory=CompletionConfig)
    sub_agents: list["AgentConfig"] | None = Field(default=None)
    state: State = Field(default_factory=State)


WEBSOCKET_URI="http://localhost:7543/v1/audio/transcriptions"
class STTConfig(AgentConfig):
    agent_base_url: str = "http://localhost:3389/v1"
    agent_token: str = "mbodi-demo-1"
    transcription_endpoint: str = "/audio/transcriptions"
    translation_endpoint: str = "/audio/translations"
    websocket_uri: str = "wss://api.mbodi.ai/audio/v1/transcriptions"
    place_holder: str = "Loading can take 30 seconds if a new model is selected..."

class TTSCConfig(AgentConfig):
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    first_speaker: str = "Luis Moray"
    second_speaker: str = "Sofia Hellen"
    first_language: str = "en"
    second_language: str = "es"
    gpu: bool = True


"""
Shared State is a dictionary that is shared between all the agents in the pipeline. It is used to store information that is needed by multiple agents.
An example of shared state is the `clear` key that is used to clear the state of all the agents in the pipeline.
"""
def persist_post_process(_prompt:str, response:str, state:State, shared_state:State | None = None) -> str:
    if shared_state.get("clear"):
        state.clear()
        return ""
    persist = state.get("persist", response)
    return response if persist in response or persist in ("No audio yet...", "Not a complete instruction") else persist


def instruct_pre_process(prompt: str, state: State, shared_state: State | None = None) -> str:
    """Either wait or forward the prompt to the agent."""
    if shared_state.get("clear"):
        state.clear()
        return ""
    if shared_state.get("instruct_status") == "wait":
        return ""
    return prompt

def instruct_post_process(prompt: str, response: str, state: State, shared_state: State | None = None) -> str:
    """Signal movement if the response is movement."""
    if shared_state.get("clear"):
        state.clear()
        return ""
    if response in ["incomplete", "noise"]:
        shared_state["actor_status"] = "wait"
        return state.get("persist", "Not a complete instruction")
    if response == "movement":
        shared_state["moving"] = True
    shared_state["instruct_status"] = "repeat"
    shared_state["actor_status"] = "ready"
    return persist_post_process(prompt, response, state, shared_state)

def actor_pre_process(prompt: str, state: State, shared_state: State | None = None) -> str:
    if shared_state.get("clear"):
        state.clear()
        return ""
    if shared_state.get("actor_status") == "wait":
        return ""
    return prompt

def actor_post_process(prompt: str, response: str, state: State, shared_state: State | None = None) -> str:
    if shared_state.get("clear"):
        state.clear()
        return ""
    shared_state["speaker_status"] = "ready"
    return persist_post_process(prompt, response, state, shared_state)


def speaker_post_process(prompt: str, response: str, state: State, shared_state: State | None = None) -> str:
    if shared_state.get("clear"):
        state.clear()
        return ""
    shared_state["speaker_status"] = "done"
    shared_state["actor_status"] = "wait"
    shared_state["instruct_status"] = "ready"
    state[""]
    persist_post_process(prompt, response, state, shared_state), 


class TranslateConfig(CompletionConfig):
    source_language: str = "en"
    target_language: str = "es"
    prompt: Callable = lambda x, y: f"Translate the following text to {x} if it is {y} and vice versa. Respond with only the translated text and nothing else."
    reminder: str = "Remember to respond with only the translated text and nothing else."

class InstructConfig(CompletionConfig):
    prompt: str = "Determine whether the following text is a command for physical movement,other actionable command, question, incomplete statement, or background noise. Respond with only ['movement', 'command', 'question', 'incomplete', 'noise']"
    reminder: str = "Remember that you should be very confident to label a command as movement. If you are not sure, label it as noise. You should be very eager to label a question as a question and a command as a command. If you are not sure, label it as incomplete."
    guidance: Guidance = Guidance(choices=["movement","command", "question", "incomplete", "noise"])
    post_process: Callable[[str, str, State], str] = route_instruct_post_process
