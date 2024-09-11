from collections.abc import Callable, Generator
import os
from typing import Annotated, Any, ParamSpec, Self, Union, dataclass_transform, overload

from gradio.components import Component
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field
from pydantic.annotated_handlers import GetCoreSchemaHandler
from pydantic.config import ConfigDict
from pydantic.json_schema import JsonSchemaValue, SkipJsonSchema
from pydantic.types import SecretStr
from pydantic_core.core_schema import CoreSchema, bool_schema, json_or_python_schema
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich import Console

WEBSOCKET_URI="http://localhost:7543/v1/audio/transcriptions"



"""
The logic of Embodied Agents V2 is not unsimilar to general concurrent programming. Every agent has a responsibility, a local state, 
a shared state, and completion configuration for querying an endpoint. The novelty of this framework
is that emphasis is placed on control flow rather than data flow. Consider the following problem statement of real-time autonomous systems:

**How can you make the best decision in the least amount of time, using only the information you have available?**

We can break this down into two subproblems:

1. Minimizng the required information to make a decision.
2. Maximizing the quality of the decision.

One can immediately notice that the problem of concurrency is quite ubiquitous. From low level multiprocessing with
context management, to handling user sessions, to desiging a RAG system, at its core, the problem is just safely
updating a **State** with multiple operators and passing as little context as possible through  **State Transition**s.

The true challenge is, as it always has been, scale. Robotics, however, is unique in this  one needs to reach truly massive levels
of synchronized concurrency for just a single embodiment. Image scheduling a zoom call between just 10
people who all need to do a certain task based on the results of everyone else's task and if one person fails,
the whole system fails. Oh and the whole group needs to do this at least 10 times a second.

You may wonder, isn't that just what normal websites are doing which is even faster and handles more processes? Yes... and no. 
The difference is the complexity or richness of the data that is being sent and operated on. It must be sent

1. Exactly correctly.
2. Fast Enough to make a cognitive decision.

A cell network doesnt use much brain power to decide what phone to send a text to. But
an embodied agent needs to think about what to do next for each individual part of the outside world,
its internal state, monitor its progress, have fail safes, and all around as fast as you are able to move your hand.

For example, a web server may have no need for a real-time shared state but an embodied collective of agents
MUST stop moving its hand before it hits the table. So instead of considering topologies or specific communication patterns like
server/client or pub/sub, we define only the FSM and the transitions. Any message passing or network protocol can
be hooked up with currently http, grpc, websockets, and ROS currently supported.

The benefit of using AgentsV2 is that every piece of data is backed by a pydantic schema and conversion methods to any
other data type you will likely need. Some of the most common include:

- Numpy, Torch, and TF Tensors
- JSON, msgpack, Protobuf
- Gym Environment
- RLDS Dataset
- ROS2 Message
- Apache Arrow Flight Compatible GRPC
- LeRobot, HuggingFace and PyArrow Tables
- VLLM MultiModal Data Input
- Gradio Input and Output


------------------------------------------------------------------------------------------------------------------------------------

The following code can be used as a boiler plate for creating a new agent. For real-time generative intelligence.

The main configuration components are:

- **Shared State**: A dictionary that stores a shared state between all the agents in the pipeline. It is used to store information that is needed by multiple agents.
- **Local State**: A dictionary that stores the state of the agent. It is takes precedence over the shared state for conflicting keys.
- **Modifiers**: A function or agent that takes the prompt, response, and state and modifies it such as filtering, persisting, or updating a state.

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
        return getattr(self, key, default)


    def keys(self):
        yield from [key for key in self.__dict__ if not key.startswith("_")]

    def _clear_state(self):
        self._cleared = True
        for key in self.keys():
            delattr(self, key)


    @property
    def clear(self) -> bool:
        """Returns a CallableBool object that evaluates the cleared state."""
        return CallableBool(self._cleared, self._clear)

    def _clear(self):
        self._cleared = True
        for key in self.keys():
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



class Guidance(Stateful):
    choices: list[str] | None = Field(default=None, examples=[["Yes", "No"]])
    json_schema: str | dict | JsonSchemaValue | None = Field(
      default=None,
      description="The json schema as a dict or string.",
      examples=[{"type": "object", "properties": {"key": {"type": "string"}}}],
    )


class BaseAgentConfig(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=False, extra="allow", arbitrary_types_allowed=True)
    base_url: str = Field(default="https://api.mbodi.ai/v1")
    auth_token: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("MBODI_API_KEY", "mbodi-demo-1")))


class CompletionConfig(Stateful):
    guidance: Guidance | None = Field(default=None, examples=[Guidance(choices=["Yes", "No"])])
    prompt_modifier: Annotated[Callable[[str, State], str] | BaseAgentConfig | None, SkipJsonSchema] = Field(default=None, description="A callable or agent that takes the prompt and state and returns a modified prompt.")
    response_modifier: Annotated[Callable[[str, str, Union[State, None]], str] | BaseAgentConfig | None, SkipJsonSchema] = Field(default=None, description="A callable or agent that takes the prompt, response, and state and returns a modified prompt.")
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
    gradio_io: Callable[[Any], tuple[Component, Component]] | None = Field(default=None, description="The input and output components for the Gradio interface.")


"""
Shared State is a dictionary that is shared between all the agents in the pipeline. It is used to store information that is needed by multiple agents.
An example of shared state is the `clear` key that is used to clear the state of all the agents in the pipeline.
"""


console = Console(style="bold yellow")

def persist_maybe_clear(_prompt:str, response:str | dict | tuple | Generator[str | dict | tuple , None, None], local_state:State, shared_state:State | None = None) -> str:
    """If the response is not a complete instruction, it will return the previous response. Useful to stabalize the response of the agent."""
    def _persist_maybe_clear(_prompt:str, response:str | dict | tuple, local_state:State, shared_state:State | None = None) -> str:
        print(f"Prompt: {_prompt}")
        console.print(f"RESPONSE: {response}, LOCAL_STATE: {local_state}, SHARED_STATE: {shared_state}")
        # return_response =  copy.deepcopy(response)
        if local_state.check_clear(shared_state):
            return ""
        if isinstance(response, tuple):
            response, new_update = response
            local_state.update(response=response, new_update=new_update)
        elif isinstance(response, dict):
            local_state.update(response)
            local_state.update(latest_response=response.get("response", response))
        

        print(f"persist in local_state: {local_state.persist}")
        persist = local_state.get("persist", response) or ""
        print(f"Persist: {persist}")
        
        final_response = response if persist in response or persist in ("No audio yet...", "Not a complete instruction") else persist
        if final_response in ("Not enough audio yet.", "Not a complete instruction"):
            shared_state.update(actor_status="wait")
            return final_response
            # if isinstance(response, dict):
            #     return_response.clear()
            #     return return_response
            # return "", "" if isinstance(return_response, tuple) else return_response
        shared_state.update(instruct_states="ready")
        return final_response
        # return final_response, new_update if isinstance(response, tuple) else final_response
    if not isinstance(response,Generator):
        return _persist_maybe_clear(_prompt, response, local_state, shared_state)
    for chunk in response:
        console.print(f"Chunk: {chunk}", style="bold green")
        yield _persist_maybe_clear(_prompt, chunk, local_state, shared_state)



class TranslateConfig(CompletionConfig):
    source_language: str = "en"
    target_language: str = "en"
    prompt: Callable = lambda x, y: f"Translate the following text to {x} if it is {y} and vice versa. Respond with only the translated text and nothing else."
    reminder: str = "Remember to respond with only the translated text and nothing else."

