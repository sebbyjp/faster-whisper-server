from typing import Generator
from mbodied.agents import Agent
from typing_extensions import override

from faster_whisper_server.agents.config import AgentConfig, State

# Decorator to enable overriding the state dict with kwargs for a signature

def maybe_override_state_dict(func):
    """Decorator to enable overriding the state dict with kwargs for a signature."""

    def wrapper(self, *args, **kwargs):
        """Wrap the function."""
        state_dict = {"local_state": self.local_state, "shared_state": self.shared_state}
        state_dict.update(kwargs)
        return func(self, *args, **state_dict)

    return wrapper

class StatefulAgent(Agent):
    """An agent that maintains state between calls."""

    def __init__(self, config: AgentConfig, shared_state: State | None = None, **kwargs) -> None:
        """Initialize the agent."""
        self.config = config
        self.local_state = State(is_first=True, is_terminal=False)
        self.shared_state = shared_state
        self.stream_config = config.stream_config
        self.completion_config = config.completion_config
        super().__init__(
            model_src="openai",
            api_key=config.auth_token, model_kwargs={"base_url":config.base_url, **kwargs})
    
    @maybe_override_state_dict
    def on_stream_resume(self,prompt: str, *__args: tuple,**__kwargs: dict) -> tuple:
        """Process the data and return the result."""
        if self.stream_config and self.stream_config.prompt_modifier:
            return self.stream_config.prompt_modifier(prompt, *__args, **__kwargs)
        return __args, __kwargs

    @maybe_override_state_dict
    def on_stream_yield(self,prompt: str, response: str, *__args: tuple, **__kwargs: dict) -> tuple:
        """Preprocess the data before sending it to the backend."""
        if self.stream_config and self.stream_config.response_modifier:
            return self.stream_config.response_modifier(prompt, response, *__args, **__kwargs)
        return __args, __kwargs

    @maybe_override_state_dict
    def on_completion_start(self, prompt: str,*__args: tuple, **__kwargs: dict) -> str:
        """Postprocess the data before returning it."""
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
        pass

    def handle_stream(self, *__args: tuple, **__kwargs: dict) -> tuple:
        """Handle the stream data."""
        return __args, __kwargs

    def handle_completion(self, *__args: tuple, **__kwargs: dict) -> tuple:
        """Handle the completion data."""
        return super().act(*__args, **__kwargs)

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
            return self.on_completion_finish(prompt, self.local_state, state=self.local_state, shared_state=self.shared_state)

        response = self.handle_completion(prompt, self.local_state, self.shared_state)
        self.local_state.last_response = response
        return self.on_completion_finish(prompt, response, state=self.local_state, shared_state=self.shared_state)

    @override
    def stream(self,prompt: str, *__args: tuple, **__kwargs: dict) -> Generator[str, None, None]:
        """Stream the data."""
        data = self.on_stream_resume(prompt, *__args, **__kwargs)
        data = self.handle_stream(data)
        data = self.on_stream_yield(prompt, data, *__args, **__kwargs)
        print(f"Data: {data}, type: {type(data)}")
        if not isinstance(data, Generator):
            return self.on_completion_finish(data)
        for chunk in data:
            print(f"Chunk: {chunk}")
            yield chunk

    def __call__(self, *__args: tuple, **__kwargs: dict) -> None:
        """Call the agent."""
        data = self.on_completion_start(*__args, **__kwargs)
        data = self.on_stream_start(data)
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
