from faster_whisper_server.agents.agent import StatefulAgent
from faster_whisper_server.agents.config import CompletionConfig
import pytest

# Test 1: Minimal input
def test_minimal_input():
    agent = StatefulAgent(CompletionConfig())
    prompt = "Minimal input test."
    result = agent.act(prompt, "default_model")
    assert result == "Handled by Superclass"

# Test 2: Maximal input
def test_maximal_input():
    agent = StatefulAgent(CompletionConfig())
    prompt = "Maximal input test."
    kwargs = {
        "state": agent.local_state,
        "shared_state": agent.shared_state,
        "extra_body": '{"choices":["command","question"]}',
        "context": {"input": "test input"},
        "additional_arg": "extra"
    }
    result = agent.act(prompt, "max_model", **kwargs)
    assert result == "Handled by Superclass"

# Test 3: Missing arguments (state missing)
def test_missing_argument():
    agent = StatefulAgent(CompletionConfig())
    prompt = "Missing argument test."
    kwargs = {
        "shared_state": agent.shared_state,
        "extra_body": '{"choices":["command","question"]}'
    }
    result = agent.act(prompt, "default_model", **kwargs)
    assert result == "Handled by Superclass"

# Test 4: Empty prompt
def test_empty_prompt():
    agent = StatefulAgent(CompletionConfig())
    prompt = ""
    result = agent.act(prompt, "default_model")
    assert result == "Handled by Superclass"

# Test 5: Invalid types
def test_invalid_types():
    agent = StatefulAgent(CompletionConfig())
    prompt = 12345  # Invalid type for the prompt (should be a string)
    with pytest.raises(TypeError):
        agent.act(prompt, "default_model")

# Test 6: State transition
def test_state_transition():
    agent = StatefulAgent(CompletionConfig())
    prompt = "State transition test."
    agent.local_state['is_first'] = False  # Simulate state transition
    result = agent.act(prompt, "transition_model")
    assert result == "Handled by Superclass"

# Test 7: Super interaction with default arguments
def test_super_interaction_default():
    agent = StatefulAgent(CompletionConfig())
    prompt = "Super interaction with default args."
    result = agent.act(prompt, "default_model")
    assert result == "Handled by Superclass"

# Test 8: Super interaction with extra kwargs
def test_super_interaction_extra_kwargs():
    agent = StatefulAgent(CompletionConfig())
    prompt = "Super interaction with extra kwargs."
    kwargs = {
        "context": {"input": "additional context"},
        "extra_arg": "extra data"
    }
    result = agent.act(prompt, "super_model", **kwargs)
    assert result == "Handled by Superclass"

# Test 9: Variadic argument handling with no extra arguments
def test_variadic_no_extra_args():
    agent = StatefulAgent(CompletionConfig())
    prompt = "Variadic test with no extra args."
    result = agent.act(prompt, "default_model")
    assert result == "Handled by Superclass"

# Test 10: Variadic argument handling with irrelevant kwargs
def test_variadic_irrelevant_kwargs():
    agent = StatefulAgent(CompletionConfig())
    prompt = "Variadic test with irrelevant kwargs."
    kwargs = {
        "irrelevant_key": "irrelevant_value",
        "state": agent.local_state
    }
    result = agent.act(prompt, "irrelevant_model", **kwargs)
    assert result == "Handled by Superclass"


if __name__ == '__main__':
    pytest.main([__file__])