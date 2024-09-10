from collections.abc import Callable
import os
from typing import ParamSpec, Self, Union, dataclass_transform, overload, Annotated, Any

from gradio.components import Component
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema
from pydantic.json_schema import JsonSchemaValue
from pydantic.types import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

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


class Stateful(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    @overload
    def update(self, **kwargs) -> Self: ...
    @overload
    def update(self, data:dict) -> Self: ...

    def update(self, data:dict | None = None, **kwargs) -> Self:
        """Update the state with the given data.

        Args:
            data (dict, optional): The data to update the state with. Defaults to None.
            kwargs: The data to update the state with as keyword arguments.

        Returns:
            Stateful: The updated state.
        """
        kwargs.update(data or {})
        for key, value in data.items():
            setattr(self, key, value)

    def clear(self) -> Self:
        for key in self.keys():
            delattr(self, key)

class State(Stateful):
    is_first: bool = Field(default=True)   
    is_terminal: bool = Field(default=False)
    wait: bool = Field(default=False)
    repeat: str | None = Field(default=None)
    persist: str | None = Field(default=None)
    last_response: str | None = Field(default=None)

class Guidance(Stateful):
    choices: list[str] | None = Field(default=None, examples=[["Yes", "No"]])
    json_schema: str | dict | JsonSchemaValue | None = Field(
      default=None,
      description="The json schema as a dict or string.",
      examples=[{"type": "object", "properties": {"key": {"type": "string"}}}],
    )


class BaseAgentConfig(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=False)
    base_url: str = "https://api.mbodi.ai/v1"
    auth_token: SecretStr = os.getenv("MBODI_API_KEY", "mbodi-demo-1")


class CompletionConfig(Stateful):
    guidance: Guidance | None = Field(default=None, examples=[Guidance(choices=["Yes", "No"])])
    prompt_modifier: Annotated[Callable[[str, State], str] | BaseAgentConfig | None, SkipJsonSchema] = Field(default=None, description="A callable or agent that takes the prompt and state and returns a modified prompt.")
    response_modifier: Annotated[Callable[[str, str, Union[State, None]], str] | BaseAgentConfig | None, SkipJsonSchema] = Field(default=None, description="A callable or agent that takes the prompt, response, and state and returns a modified prompt.")
    reminder: str | None = Field(default=None, examples=["Remember to respond with only the translated text and nothing else."])

@dataclass_transform(order_default=False)
class AgentConfig(BaseAgentConfig, ParamSpec):
    base_url: str = "https://api.mbodi.ai/v1"
    auth_token: str = "mbodi-demo-1"
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

def persist_maybe_clear(_prompt:str, response:str | dict | tuple, local_state:State, shared_state:State | None = None) -> str:
    """Useful to stabalize the response of the agent.

    Convenience conventions for the response:
    - Always assumes a tuple response is (response, state).
    - A dict has "state" and "response" keys.
    """
    if shared_state.get("clear"):
        shared_state.clear(), local_state.clear()
        return ""
    if isinstance(response, tuple):
        response, new_update = response
        local_state.update(new_update)
    elif isinstance(response, dict):
        local_state.update(response.get("state", {}))
        local_state.update(latest_response=response.get("response", response))

        
    persist = local_state.get("persist", response)
    return response if persist in response or persist in ("No audio yet...", "Not a complete instruction") else persist




class TranslateConfig(CompletionConfig):
    source_language: str = "en"
    target_language: str = "en"
    prompt: Callable = lambda x, y: f"Translate the following text to {x} if it is {y} and vice versa. Respond with only the translated text and nothing else."
    reminder: str = "Remember to respond with only the translated text and nothing else."

