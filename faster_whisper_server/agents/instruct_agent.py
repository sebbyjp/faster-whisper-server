from typing import Text
import gradio as gr

from faster_whisper_server.agents.agent import StatefulAgent
from faster_whisper_server.agents.config import AgentConfig, CompletionConfig, Guidance, State, persist_maybe_clear

INSTRUCT_CHOICES = ["movement", "command", "question", "incomplete", "noise"]


def instruct_pre_process(prompt: str, local_state: State, shared_state: State | None = None) -> str:
    """Either wait or forward the prompt to the agent."""
    if "stop movement" in prompt.lower() or "halt" in prompt.lower():
        shared_state.clear()

    if local_state.check_clear(shared_state) or shared_state.get("actor_status") == "wait":
        return ""
    return prompt

def instruct_post_process(prompt: str, response: str, local_state: State, shared_state: State | None = None) -> str:
    """Signal movement if the response is movement."""
    if response in ["incomplete", "noise"]:
        shared_state.update(actor_status="wait")    
        return local_state.get("persist", "Not a complete instruction")
    if response == "movement":
        shared_state.update(moving=True)
    if response == "question":
        shared_state.update(actor_status="ready")
        shared_state.update(instruct_status="repeat")

    shared_state.update(shared_state)
    return persist_maybe_clear(prompt, response, local_state, shared_state)


instruct_completion =CompletionConfig(
    prompt_modifier= lambda prompt, local_state, shared_state: f"""
        Determine whether the following text after "INPUT:" is one of the following:
            movement: a command for physical movement
            command: other actionable command
            question: a question to be answered
            incomplete statement: a statement that does not need a response
            noise: irrelevant text

        Respond with only ['movement', 'command', 'question', 'incomplete', 'noise']

        INPUT: {prompt}
    """,
    response_modifier= instruct_post_process,
    reminder= """
        Remember that you should be very confident to label a command as movement.
        If you are not sure, label it as noise. You should be very eager to label a question as a question and a command as a command.
        If you are not sure, label it as incomplete.""",

    guidance= Guidance(guided_choice=INSTRUCT_CHOICES),
)


instruct_config = AgentConfig(
    base_url="https://api.mbodi.ai/v1",
    auth_token="mbodi-demo-1", # noqa: S106
    system_prompt="""You are a world class instruction-detector. You always determine the relevance of a potential instruction.""",
    completion_config=instruct_completion,
    model="astroworld",
)


class InstructAgent(StatefulAgent[Text, Text, State, AgentConfig]):
    pass
