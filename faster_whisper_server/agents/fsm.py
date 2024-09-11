from faster_whisper_server.agents.actor import actor_post_process, actor_pre_process
from faster_whisper_server.agents.agent import State, StatefulAgent, AgentConfig
from faster_whisper_server.agents.instruct_agent import InstructAgent, instruct_config
from faster_whisper_server.agents.speaker_agent import SpeakerAgent, speaker_config
from faster_whisper_server.agents.whisper_agent import WhisperAgent, whisper_config

state = State(is_first=True, is_terminal=False)
whisper_agent = WhisperAgent(state=state)

class Actor(StatefulAgent):
    def __init__(self, config: AgentConfig, shared_state: State):
        super().__init__(config=config, state=state)


    def on_completion_start(self, prompt: str, response: str, *__args, **__kwargs) -> str:
        return actor_post_process(prompt, response, self.local_state, self.shared_state)
    


    def pre_process(self, data):
        return actor_pre_process(data)

    def run(self, data):
        return self.post_process(self.pre_process(data))



demo = whisper_agent.demo()
with demo as demo:
    demo.launch(
    server_name="0.0.0.0",
    server_port=7861,
    share=False,
    debug=True,
    show_error=True,
    )


