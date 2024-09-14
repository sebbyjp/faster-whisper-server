from faster_whisper_server.agents.agent import StatefulAgent, State
from faster_whisper_server.agents.config import AgentConfig, CompletionConfig, Guidance
import pytest

# Fixing the issue by updating attributes within the Guidance object instead of replacing it with a dict

def test_run_final():
    # Test 1: Test Stateful update
    state = State()
    state.is_first = True
    state.is_terminal = False
    updated_state = state.update(data={"is_first": False})
    assert updated_state.is_first is False
    updated_state = state.update(is_terminal=True)
    assert updated_state.is_terminal is True

    # Test 2: Test Stateful get()
    assert state.get("is_first") is False
    assert state.get("non_existent_key", default="default_value") == "default_value"

    # Test 3: Test Stateful clear and check_clear
    other_state = State()
    state._clear()
    assert state._cleared is True
    other_state._cleared = True
    assert state.check_clear(other_state) is True
    assert state._cleared is True

    # Test 4: Test CompletionConfig modifiers
    def mock_prompt_modifier(prompt: str, state: State) -> str:
        return f"Modified {prompt}"

    def mock_response_modifier(prompt: str, response: str, state: State | None) -> str:
        return f"Modified {response}"

    completion_config = CompletionConfig()
    completion_config.prompt_modifier = mock_prompt_modifier
    completion_config.response_modifier = mock_response_modifier

    modified_prompt = completion_config.prompt_modifier("Test Prompt", state)
    assert modified_prompt == "Modified Test Prompt"

    modified_response = completion_config.response_modifier("Test Prompt", "Test Response", state)
    assert modified_response == "Modified Test Response"

    # Test 5: Test AgentConfig state and sub_agents
    sub_agent = AgentConfig()
    sub_agent.system_prompt = "Sub agent prompt"
    
    agent = AgentConfig()
    agent.system_prompt = "Main agent prompt"
    agent.sub_agents = [sub_agent]
    
    assert agent.system_prompt == "Main agent prompt"
    assert len(agent.sub_agents) == 1
    assert agent.sub_agents[0].system_prompt == "Sub agent prompt"
    assert agent.state.is_first is True
    assert agent.state.is_terminal is False

    # Test 6: Test CompletionConfig guidance updates
    guidance = Guidance()
    guidance.guided_choice = ["Yes", "No"]
    completion_config = CompletionConfig()
    completion_config.guidance = guidance
    assert completion_config.guidance.guided_choice == ["Yes", "No"]

    # Update guidance guided_choice directly without replacing the object
    completion_config.guidance.guided_choice = ["Maybe"]
    assert completion_config.guidance.guided_choice == ["Maybe"]

    # Test 7: Test AgentConfig state interaction
    agent = AgentConfig()
    assert agent.state.is_first is True
    assert agent.state.is_terminal is False
    agent.state.update(is_first=False, last_response="Finished")
    assert agent.state.is_first is False
    assert agent.state.last_response == "Finished"

    # Test 8: Test BaseAgentConfig with CompletionConfig
    completion_config = CompletionConfig()
    completion_config.prompt_modifier = "Modifier"
    completion_config.reminder = "This is a reminder"
    
    agent = AgentConfig()
    agent.completion_config = completion_config
    agent.system_prompt = "Main agent prompt"
    
    assert agent.completion_config.prompt_modifier == "Modifier"
    assert agent.completion_config.reminder == "This is a reminder"

    return "All tests passed!"


if __name__ == '__main__':
    pytest.main([__file__])