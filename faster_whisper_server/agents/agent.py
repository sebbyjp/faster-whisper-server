from collections.abc import Generator
from inspect import signature
from typing import ClassVar, Generic, TypeVar

from gradio import Text
from gradio.components import Component as GRComponent
from mbodied.agents.language.language_agent import LanguageAgent
from typing_extensions import override  # noqa: UP035

from faster_whisper_server.agents.config import AgentConfig, State
from faster_whisper_server.agents.utils import filter_function_params


def maybe_override_state_dict(func):
    """Decorator to enable overriding the state dict with kwargs for a signature."""
    params = signature(func).parameters
    print(f"Func: {params}")

    def wrapper(self: "StatefulAgent", *args: tuple, **kwargs: dict) -> tuple:
        state = self.local_state
        shared_state = self.shared_state
        print(f"Calling {func.__name__} with {args} and {kwargs}")
        return func(self, *args, local_state=state, shared_state=shared_state)

    return wrapper



class Prompts:
    """A class to hold the prompts for the agent."""

    system: str | None = None
    completion: str | None = None
    reminder: str | None = None
    guidance: str | None = None
    examples: str | None = None

ObsT, StateT, ActionT, ConfigT = (
    TypeVar("ObsT", bound=GRComponent),
    TypeVar("StateT", bound=State),
    TypeVar("ActionT", bound=GRComponent),
    TypeVar("ConfigT", bound=AgentConfig),
)


class StatefulAgent(LanguageAgent, Generic[ObsT, ActionT, StateT, ConfigT]):
    """An agent that maintains state between calls.

    Usage: `agent = StatefulAgent[ObsT, ActionT, StateT, ConfigT](config, shared_state)`
    """

    _obs_type: ClassVar[type[ObsT]] = GRComponent
    _action_type: ClassVar[type[ActionT]] = GRComponent
    _state_type: ClassVar[type[StateT]] = State
    _config_type: ClassVar[type[ConfigT]] = AgentConfig

    def __init__(self, config: ConfigT, shared_state: State | None = None, **kwargs) -> None:
        """Initialize the agent."""
        self.config = config
        self.local_state = self._state_type(is_first=True, is_terminal=False)
        self.shared_state = shared_state
        self.stream_config = config.stream_config
        self.completion_config = config.completion_config
        self.io: tuple[ObsT, ActionT] = self._obs_type, self._action_type
        print(f"{self.__class__.__name__}, base_url: {self.config.base_url}, auth_token: {self.config.auth_token}")
        super().__init__(
            model_src="openai", api_key=config.auth_token, model_kwargs={"base_url": config.base_url, **kwargs}
        )

    def __class_getitem__(
        cls, items: tuple[type[ObsT], type[ActionT], type[StateT], type[ConfigT]] | None
    ) -> type["StatefulAgent"]:
        if len(items) == 3:
            items = (items[0], items[1], items[2], AgentConfig)
        elif len(items) == 2:
            items = (items[0], items[1], State, items)
        elif len(items) == 1:
            items = (items[0], Text, State, AgentConfig)
        elif len(items) == 0:
            items = (Text, Text, State, AgentConfig)
        obs, action, state, config = items
        cls._obs_type = obs
        cls._action_type = action
        cls._state_type = state
        cls._config_type = config
        return cls

    @maybe_override_state_dict
    def on_stream_resume(self, prompt: str, *__args: tuple, **__kwargs: dict) -> tuple:
        """Process the data and return the result."""
        print(f"Prompt: {prompt}, args: {__args}, kwargs: {__kwargs}")
        if self.stream_config and self.stream_config.prompt_modifier:
            return self.stream_config.prompt_modifier(prompt, *__args, **__kwargs)
        return __args, __kwargs

    @maybe_override_state_dict
    def on_stream_yield(self, prompt: str, response: str, *__args: tuple, **__kwargs: dict) -> tuple:
        """Preprocess the data before sending it to the backend."""
        if self.stream_config and self.stream_config.response_modifier:
            return self.stream_config.response_modifier(prompt, response, *__args, **__kwargs)
        return __args, __kwargs

    @maybe_override_state_dict
    def on_completion_start(self, prompt: str, *__args: tuple, **__kwargs: dict) -> str:
        """Preprocess the prompt before sending it to the backend."""
        if self.completion_config and self.completion_config.prompt_modifier:
            return self.completion_config.prompt_modifier(prompt, *__args, **__kwargs)
        return prompt

    @maybe_override_state_dict
    def on_completion_finish(self, prompt: str, response: str, *__args: tuple, **__kwargs: dict) -> str:
        """Postprocess the data before returning it."""
        if self.completion_config and self.completion_config.response_modifier:
            return self.completion_config.response_modifier(prompt, response, *__args, **__kwargs)
        return prompt

    def reset(self) -> None:
        """Reset the agent."""
        self.local_state = State(is_first=True, is_terminal=False, repeat=None, wait=False, persist=None)

    def close(self) -> None:
        """Close the agent."""

    @maybe_override_state_dict
    def handle_stream(self, *__args: tuple, **__kwargs: dict) -> tuple:
        """Handle the stream data."""
        args, kwargs  = filter_function_params(super().act_and_stream, *__args, **__kwargs)
        if self.stream_config and self.stream_config.guidance:
            kwargs.update(extra_body=self.stream_config.guidance.model_dump(mode="json", exclude_none=True),
            model = self.config.model)
        kwargs.pop("local_state", None), kwargs.pop("shared_state", None)
        print(f"Calling {super().__class__.__name__} with {args} and {kwargs}")
        return super().act_and_stream(*args, **kwargs)

    @maybe_override_state_dict
    def handle_completion(self, *__args: tuple, **__kwargs: dict) -> tuple:
        """Handle the completion data."""
        args, kwargs  = filter_function_params(super().act_and_stream, *__args, **__kwargs)
        if self.completion_config and self.completion_config.guidance:
            kwargs.update(extra_body=self.completion_config.guidance.model_dump(mode="json", exclude_none=True),
            model = self.config.model)
        print(f"Calling {super().__class__.__name__} with {args} and {kwargs}")
        kwargs.pop("local_state", None), kwargs.pop("shared_state", None)

        return super().act(*args,**kwargs)

    def __del__(self) -> None:
        """Delete the agent."""
        self.close()

    def __enter__(self) -> "StatefulAgent":
        """Enter the agent."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Exit the agent."""
        self.close()

    @override
    def act(self, *args, **kwargs) -> None:
        """Act on the data."""
        if self.local_state.wait:
            return ""
        prompt = self.on_completion_start(*args, **kwargs, state=self.local_state, shared_state=self.shared_state)
        if self.local_state.repeat:
            return self.on_completion_finish(
                prompt, self.local_state, state=self.local_state, shared_state=self.shared_state
            )

        response = self.handle_completion(prompt, self.local_state, self.shared_state)
        self.local_state.last_response = response
        return self.on_completion_finish(prompt, response, state=self.local_state, shared_state=self.shared_state)

    @override
    def stream(self, prompt: str, *__args: tuple, **__kwargs: dict) -> Generator[str, None, None]:
        """Stream the data."""
        data = self.on_stream_resume(prompt, *__args, **__kwargs)
        data = self.handle_stream(data)
        data = self.on_stream_yield(prompt, data, *__args, **__kwargs)
        print(f"Data: {data}, type: {type(data)}")
        if not isinstance(data, Generator):
            return self.on_completion_finish(prompt, data, *__args, **__kwargs)
        for chunk in data:
            print(f"Chunk: {chunk}")
            yield chunk

    def __call__(self, *__args: tuple, **__kwargs: dict) -> None:
        """Call the agent."""
        data = self.on_completion_start(*__args, **__kwargs)
        data = self.on_stream_resume(data)
        data = self.act(data)
        data = self.on_stream_yield(data)
        data = self.on_completion_finish(data)

    def __repr__(self) -> str:
        """Return a string representation of the agent."""
        return f"{self.__class__.__name__}(config={self.config})"

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return repr(self)

    def __hash__(self) -> int:
        """Return the hash of the agent."""
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        """Return whether the agent is equal to another object."""
        return isinstance(other, self.__class__) and hash
