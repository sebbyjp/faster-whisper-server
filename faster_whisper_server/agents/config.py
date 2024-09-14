from collections.abc import Callable, Generator
import os
from typing import Annotated, Any, Final, Literal, NamedTuple, ParamSpec, Self, Union, dataclass_transform, overload

from gradio.components import Component
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_serializer, model_validator
from pydantic.annotated_handlers import GetCoreSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic.types import SecretStr
from pydantic_core.core_schema import CoreSchema, bool_schema, json_or_python_schema
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console

WEBSOCKET_URI="http://localhost:7543/v1/audio/transcriptions"



"""
# Embodied Agents V2: Concurrent Programming for Real-Time Autonomous Systems

Embodied Agents V2 applies concurrent programming principles to real-time autonomous systems, emphasizing control flow over data flow. This framework addresses two key challenges:

1. Minimizing the information required for decision-making
2. Maximizing decision quality

## Core Concepts

Each agent in the system has:

- • A specific responsibility
- • A local state
- • Access to a shared state
- • A completion configuration for querying endpoints

## Concurrency in Robotics

Robotics demands high levels of synchronized concurrency. Unlike simple network operations, embodied agents must:

- • Process complex, rich data
- • Make cognitive decisions rapidly
- • Coordinate multiple system components in real-time

## Framework Design

Instead of focusing on specific communication patterns, Embodied Agents V2 defines:

- • Finite State Machines (FSM)
- • State transitions

This approach allows flexibility in message passing and network protocols, supporting HTTP, gRPC, WebSockets, and ROS.

## Data Handling

AgentsV2 uses Pydantic schemas for all data, offering conversion methods to various formats including:

- • Numpy, Torch, and TensorFlow tensors
- • JSON, msgpack, Protobuf
- • Gym Environment
- • RLDS Dataset
- • ROS2 Messages
- • Apache Arrow Flight Compatible gRPC
- • LeRobot, HuggingFace, and PyArrow Tables
- • VLLM MultiModal Data Input
- • Gradio Input and Output

## Key Components

The main configuration components include:

- • Shared State: A dictionary for information shared across agents
- • Local State: Agent-specific state information
- • Modifiers: Functions or agents that modify prompts, responses, or states

## Implementation

The provided code outlines the core classes and configurations for implementing Embodied Agents V2:

- • Stateful and State classes for managing agent states
- • Guidance class for providing decision-making parameters
- • BaseAgentConfig and AgentConfig for setting up agent configurations
- • CompletionConfig for specifying how agents interact with endpoints


------------------------------------------------------------------------------------------------------------------------------------
    The Decorator (maybe_override_state_dict)

    The decorator is responsible for ensuring that every method that uses it (like on_stream_resume, on_stream_yield, on_completion_start, on_completion_finish) automatically injects the agent’s local_state and shared_state as keyword arguments into the function. This means the decorator abstracts away the need for the developer to manually manage state injection in these methods.

    Key Benefits of the Decorator:

        •	State management abstraction: It simplifies the methods that require state by automatically passing local_state and shared_state.
        •	Centralized state injection: Makes it easy to inject shared logic around state handling without repeating it for each method.
        •	Separation of concerns: Keeps the method signatures clean, focusing on the task rather than managing the state.

    The Hooks (on_stream_resume, on_stream_yield, on_completion_start, on_completion_finish)

    These methods act as hooks that allow you to customize or override the behavior at different stages of the agent’s lifecycle, such as:

        •	Before sending the prompt (on_completion_start).
        •	After receiving a response (on_completion_finish).
        •	During stream handling (on_stream_resume, on_stream_yield).

    Each hook has specific logic, such as modifying prompts or responses using the provided CompletionConfig, based on runtime configurations like prompt_modifier or response_modifier.

    Key Benefits of Hooks:

        •	Customization: These hooks enable behavior modification at specific points in the agent’s interaction cycle.
        •	Dynamic logic: Based on runtime configurations (prompt_modifier, response_modifier), you can dynamically alter the flow without modifying the base logic of the agent.

    Are Both Necessary?

    The decorator and the hooks serve complementary but distinct purposes:

        1.	The Decorator: Abstracts state handling, reducing code duplication and ensuring consistency across all methods where the state is required.
        2.	The Hooks: Provide a way to customize or modify behavior at key points of the agent’s lifecycle. They process data (like prompts or responses) based on configuration and user inputs.

    If you only need automatic state injection and your logic for modifying prompts or responses is static, you could potentially handle everything with just the decorator. But if you need dynamic behavior modification (e.g., handling different prompt modifications based on configuration), the hooks are essential.

"""

def Identity(x):
    return x

class CallableBool(int):
    def __init__(self,  _bool: bool = False, func: Callable = Identity):
        self.func = func
        self._bool = int(_bool)

    def __bool__(self):
        return bool(self._bool)
    def __call__(self):
        return self.func()
    
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source: type[BaseModel], handler: GetCoreSchemaHandler, /) -> CoreSchema:
        """Hook into generating the model's CoreSchema.

        Args:
            source: The class we are generating a schema for.
                This will generally be the same as the `cls` argument if this is a classmethod.
            handler: A callable that calls into Pydantic's internal CoreSchema generation logic.

        Returns:
            A `pydantic-core` `CoreSchema`.
        """
        return handler(int)



class CallableBool:
    def __init__(self, _bool: bool = False, func: Callable = lambda: None):
        self.func = func
        self._bool = _bool

    def __bool__(self):
        return self._bool

    def __call__(self):
        return self.func()

    def __int__(self):
        return int(self._bool)


    @property
    def __pydantic_core_schema__(self) -> CoreSchema:
        return bool_schema()
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Callable[[Any], CoreSchema],
    ) -> CoreSchema:

        return json_or_python_schema(
                python_schema=bool_schema(),
                json_schema={"type": "boolean"},
            )

    @classmethod
    def __get_json_core_schema__(cls, handler: GetCoreSchemaHandler, /) -> JsonSchemaValue:
        return handler(bool)


class Stateful(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    _cleared: bool = PrivateAttr(default=False)

    @overload
    def update(self, **kwargs) -> Self: ...

    @overload
    def update(self, data: dict) -> Self: ...

    def update(self, data: dict | None = None, **kwargs) -> Self:
        """Update the state with the given data.

        Args:
            data (dict, optional): The data to update the state with. Defaults to None.
            kwargs: The data to update the state with as keyword arguments.

        Returns:
            Stateful: The updated state.
        """
        kwargs.update(data or {})
        for key, value in kwargs.items():
            setattr(self, key, value)

        return self

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.keys():
            return getattr(self, key)
        return default

    def keys(self):
        yield from [key for key in self.__dict__ if not key.startswith("_")]

    def _clear_state(self):
        self._cleared = True
        for key in self.keys():
            delattr(self, key)

    def __setitem___(self, key, value) -> None:
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, not {type(key).__name__}")
        setattr(self, key, value)
        super().__setattr__(key, value)


    @property
    def clear(self) -> bool:
        """Returns a CallableBool object that evaluates the cleared state."""
        return CallableBool(self._cleared, self._clear)

    def _clear(self) -> None:
        self._cleared = True
        for key in self.keys():
            if not key.startswith("_") and key not in ("model_config", "clear"):
                delattr(self, key)
        self._cleared = True

    def check_clear(self, other: "Stateful") -> bool:
        """Check if other state is cleared, clear this state if it is, and return true if the state is cleared."""
        if bool(other.clear):  # Ensure clear is evaluated as a boolean
            self.clear()  # Call clear method if needed
            return True
        return False

class State(Stateful):
    is_first: bool = Field(default=True)
    is_terminal: bool = Field(default=False)
    wait: bool = Field(default=False)
    repeat: str | None = Field(default=None)
    persist: str | None = Field(default=None)
    last_response: str | None = Field(default=None)
    clear: Any


    def __setitem___(self, key, value) -> None:
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, not {type(key).__name__}")
        setattr(self, key, value)
        super().__setattr__(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.keys():
            ret = getattr(self, key)
            return ret if ret is not None else default
        return default

class Guidance(Stateful):
    guided_choice: list[str] | None = Field(default=None, examples=[["Yes", "No"]])
    guided_json: str | dict | JsonSchemaValue | None = Field(
      default=None,
      description="The json schema as a dict or string.",
      examples=[{"type": "object", "properties": {"key": {"type": "string"}}}],
    )

class BaseAgentConfig(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=False, extra="allow", arbitrary_types_allowed=True)
    base_url: str = Field(default="https://api.mbodi.ai/v1")
    auth_token: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("MBODI_API_KEY", "mbodi-demo-1")))


class CompletionConfig(Stateful):
    guidance: Guidance | None = Field(default=None, examples=[Guidance(guided_choice=["Yes", "No"])])
    prompt_modifier: str | Callable[[str, State], str] | BaseAgentConfig | None= Field(default=None, description="A callable or agent that takes the prompt and state and returns a modified prompt.")
    response_modifier: str | Callable[[str, str, State | None], str] | BaseAgentConfig | None = Field(default=None, description="A callable or agent that takes the prompt, response, and state and returns a modified prompt.")
    reminder: str | None = Field(default=None, examples=["Remember to respond with only the translated text and nothing else."])

class AgentConfig(BaseAgentConfig):
    base_url: str = Field(default="https://api.mbodi.ai/v1")
    auth_token: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("MBODI_API_KEY", "mbodi-demo-1")))
    model: str | None = Field(default=None)
    system_prompt: str | None = Field(default=None)
    completion_config: CompletionConfig = Field(default_factory=CompletionConfig)
    stream_config: CompletionConfig | None = Field(default_factory=CompletionConfig)
    sub_agents: list["AgentConfig"] | None = Field(default=None)
    state: State = Field(default_factory=State)
    io: Callable[[Any], tuple[Component, Component]] | None = Field(default=None, description="The input and output components for the Gradio interface.")


"""
Shared State is a dictionary that is shared between all the agents in the pipeline. It is used to store information that is needed by multiple agents.
An example of shared state is the `clear` key that is used to clear the state of all the agents in the pipeline.
"""


console = Console(style="bold yellow")

def persist_maybe_clear(_prompt:str, response:str | dict | tuple | Generator[str | dict | tuple , None, None], local_state:State, shared_state:State | None = None) -> str:
    """If the response is not a complete instruction, it will return the previous response. Useful to stabalize the response of the agent."""

    def _persist_maybe_clear(_prompt:str, response:str | dict | tuple, local_state:State, shared_state:State | None = None) -> str:
        if local_state.check_clear(shared_state):
            return ""
        if isinstance(response, tuple):
            text_response, other = response
            persist, persist_other = local_state.get("last_response", response)
        should_persist = persist in text_response or persist not in ("No audio yet...", "Not a complete instruction")
        if should_persist:
            local_state.update(last_response=(text_response, other))
            return response
        local_state.update(last_response=response)
        return persist[0] + " " + text_response, other


    if not isinstance(response,Generator):
        return response
        return _persist_maybe_clear(_prompt, response, local_state, shared_state)
    previous_response = local_state.get("last_response", "")

    for chunk in response:
        console.print(f"Chunk: {str(chunk)}", style="bold green")
        if isinstance(chunk, tuple) and not any(i is None for i in chunk):
            previous_response, other = previous_response + " " + chunk[0], chunk[1]
            print(f"Previous Response: {previous_response}")
            print(f"chunk: {chunk}, other: {other}")
            persist, persist_other = local_state.get("last_response", chunk)
            chunk = (persist[0] + " " + chunk[0], other)
            local_state.update(last_response=chunk)
        # committed_response = _persist_maybe_clear(_prompt, chunk, local_state, shared_state)
        yield chunk
    return



class TranslateConfig(CompletionConfig):
    source_language: str = "en"
    target_language: str = "en"
    prompt: Callable = lambda x, y: f"Translate the following text to {x} if it is {y} and vice versa. Respond with only the translated text and nothing else."
    reminder: str = "Remember to respond with only the translated text and nothing else."

