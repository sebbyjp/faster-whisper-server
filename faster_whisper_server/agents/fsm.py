from typing import Final

import gradio as gr
from gradio import Audio, Text

from faster_whisper_server.agents.actor import actor_config
from faster_whisper_server.agents.agent import AgentConfig, State, StatefulAgent
from faster_whisper_server.agents.instruct_agent import InstructAgent, instruct_config

# from faster_whisper_server.agents.speaker_agent import Language, Speaker, SpeakerAgent, SpeakerConfig, speaker_config
from faster_whisper_server.agents.whisper_agent import WhisperAgent, WhisperConfig, whisper_config
from faster_whisper_server.colors import mbodi_color

state = State(is_first=True, is_terminal=False)
whisper_agent = WhisperAgent[Audio, Text, State, WhisperConfig](config=whisper_config, shared_state=state)
instruct_agent = InstructAgent[Text, Text, State, AgentConfig](config=instruct_config, shared_state=state)
# speaker_agent = SpeakerAgent[Text, Audio, State, SpeakerConfig](config=speaker_config, shared_state=state)
actor_agent = StatefulAgent[Text, Text, State, AgentConfig](config=actor_config, shared_state=state)

inputs: Final = {
    "instruction": lambda: gr.Textbox(label="Instruction", render=True),
    "button": lambda: gr.Button(value="Clear State", variant="primary"),
    # "first_speaker_name": speaker_agent.config.gradio_io,
    "audio": lambda:  gr.Audio(label="Audio",streaming=True),
}

outputs: Final = {
    "assistant":lambda: gr.Audio(label="Assistant",streaming=True),
    "transcription":lambda: gr.Textbox(label="Transcription", render=True),
    "instruction":lambda: gr.Textbox(label="Instruction", render=True),
    "response":lambda: gr.Textbox(label="Response", render=True),
}


with gr.Blocks(
    title="Assistant",
    theme=gr.themes.Soft(
        primary_hue=mbodi_color,
        secondary_hue="stone",
    ),
) as demo:
    with gr.Row():
        audio_in: gr.Audio = inputs["audio"]()
        audio_out: gr.Audio = outputs["assistant"]()
    with gr.Row():
        button: gr.Button = inputs["button"]()
    with gr.Row():
        with gr.Column():
            first_speaker_name = gr.Dropdown(
                label="First Speaker Name",
                # choices=get_args(Speaker),
                # value=speaker_agent.config.first_speaker,
                interactive=True,
            )
            first_speaker_language = gr.Dropdown(
                # choices=get_args(Language),
                label="First Speaker Language",
                # value=speaker_agent.config.first_language,
                interactive=True,
            )
        with gr.Column():
            with gr.Row():
                transcription: gr.Text = outputs["transcription"]()
                tps = gr.Text(label="TPS", render=True)
            instruction: gr.Text = outputs["instruction"]()
        with gr.Column():
            response: gr.Text = outputs["response"]()

        audio_in.stream(
            fn=whisper_agent.stream,
            inputs=[
                audio_in,
            ],
            outputs=[transcription, tps],
        )
        transcription.change(
            instruct_agent.act,
            inputs=[transcription],
            outputs=[instruction],
        )
        instruction.change(
            actor_agent.stream,
            inputs=[instruction],
            outputs=[response],
        )
        # response.change(
        #     speaker_agent.stream,
        #     inputs=[response, first_speaker_name, first_speaker_language],
        #     outputs=[audio_out],
        #     trigger_mode="always_last",
        # )
# demo = gr.Interface(
#     fn=whisper_agent.stream,
#     inputs=whisper_agent.config.gradio_io(),
#     outputs=whisper_agent.config.gradio_io(),
#     title="Whisper",
#     description="Stream audio to the server for transcription.",
#     live=True,
#     theme=gr.themes.Soft(
#         primary_hue=mbodi_color,
#         secondary_hue="stone",
#     ),
# )

with demo as demo:
    demo.launch(
    server_name="0.0.0.0",
    server_port=7861,
    share=False,
    debug=True,
    show_error=True,
    )


