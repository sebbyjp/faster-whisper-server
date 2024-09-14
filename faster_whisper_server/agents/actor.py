from faster_whisper_server.agents.config import AgentConfig, CompletionConfig, State, persist_maybe_clear


def actor_pre_process(prompt: str, local_state: State, shared_state: State | None = None) -> str:
    if local_state.check_clear(shared_state) or shared_state.get("actor_status") == "wait":
        return ""
    return prompt

def actor_post_process(prompt: str, response: str, local_state: State, shared_state: State | None = None) -> str:
    if local_state.check_clear(shared_state):
        return ""
    shared_state.update(actor_status="repeat")
    shared_state.update(speaker_status="ready")


    return persist_maybe_clear(prompt, response, local_state, shared_state)


actor_config = AgentConfig(
    base_url="https://api.mbodi.ai/v1",
    auth_token="mbodi-demo-1", # noqa: S106
    completion_config=CompletionConfig(
        response_modifier= actor_post_process,
        prompt_modifier= actor_pre_process,
    ),
)