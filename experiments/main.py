import asyncio
import json
import zenoh
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anyio
import gradio as gr

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class State(BaseModel):
    audio_state: dict = {}
    video_state: dict = {}
    world_state: dict = {}
    llm_state: dict = {}

async def state_manager():
    state = State()
    send_stream, receive_stream = anyio.create_memory_object_stream(max_buffer_size=10)
    
    async def update_state():
        async for update in receive_stream:
            if 'audio' in update:
                state.audio_state.update(update['audio'])
            if 'video' in update:
                state.video_state.update(update['video'])
            if 'world' in update:
                state.world_state.update(update['world'])
            if 'llm' in update:
                state.llm_state.update(update['llm'])
    
    return state, send_stream, update_state

# Global state and stream objects
global_state, global_state_send_stream, update_state_task = asyncio.run(state_manager())

# Zenoh session
zenoh_session = None

@app.on_event("startup")
async def startup_event():
    global zenoh_session
    zenoh_session = await zenoh.open()
    asyncio.create_task(update_state_task())
    asyncio.create_task(zenoh_subscriber())
    asyncio.create_task(llm_process())

@app.on_event("shutdown")
async def shutdown_event():
    if zenoh_session:
        await zenoh_session.close()

@app.get("/state")
async def get_state():
    return global_state.dict()

async def zenoh_subscriber():
    sub = await zenoh_session.declare_subscriber("robot/data", zenoh_callback)

async def zenoh_callback(sample):
    data = json.loads(sample.payload.decode("utf-8"))
    if "audio" in data:
        await process_audio(data["audio"])
    elif "video" in data:
        await process_video(data["video"])

async def process_audio(audio_data):
    audio_result = {"transcription": f"Processed: {audio_data}"}
    await global_state_send_stream.send({'audio': audio_result})

async def process_video(video_data):
    video_result = {"pose": f"Estimated pose from: {video_data}"}
    await global_state_send_stream.send({'video': video_result})

async def llm_process():
    while True:
        llm_result = {"response": f"LLM processed: {global_state.audio_state} and {global_state.video_state}"}
        await global_state_send_stream.send({'llm': llm_result})
        await zenoh_session.put("robot/llm_output", json.dumps(llm_result))
        await asyncio.sleep(1)  # Adjust as needed

# Gradio Interface
def update_ui():
    return (
        json.dumps(global_state.audio_state, indent=2),
        json.dumps(global_state.video_state, indent=2),
        json.dumps(global_state.world_state, indent=2),
        json.dumps(global_state.llm_state, indent=2)
    )

def send_command(command):
    asyncio.create_task(zenoh_session.put("robot/command", command))
    return f"Sent command: {command}"

with gr.Blocks() as demo:
    gr.Markdown("# Robot Control and Monitoring")
    with gr.Row():
        audio_state = gr.JSON(label="Audio State")
        video_state = gr.JSON(label="Video State")
    with gr.Row():
        world_state = gr.JSON(label="World State")
        llm_state = gr.JSON(label="LLM State")
    command_input = gr.Textbox(label="Send Command to Robot")
    command_output = gr.Textbox(label="Command Status")
    update_button = gr.Button("Update States")
    send_button = gr.Button("Send Command")

    update_button.click(update_ui, outputs=[audio_state, video_state, world_state, llm_state])
    send_button.click(send_command, inputs=command_input, outputs=command_output)

# Mount Gradio app
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)