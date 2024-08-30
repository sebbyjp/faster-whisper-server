import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

class State:
    def __init__(self):
        self.audio_state = {}
        self.video_state = {}
        self.world_state = {}
        self.llm_state = {}

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

async def process_audio(state, state_send_stream):
    while True:
        # Process audio
        audio_result = await process_audio_chunk()
        await state_send_stream.send({'audio': audio_result})
        # Use state if needed
        # current_state = state.audio_state

async def process_video(state, state_send_stream):
    while True:
        # Process video
        video_result = await process_video_frame()
        await state_send_stream.send({'video': video_result})
        # Use state if needed
        # current_state = state.video_state

async def llm_process(state, state_send_stream):
    while True:
        # LLM processing
        llm_result = await run_llm(state.audio_state, state.video_state, state.world_state)
        await state_send_stream.send({'llm': llm_result})
        # Update world state based on LLM output
        await state_send_stream.send({'world': compute_world_update(llm_result)})

async def main():
    state, state_send_stream, update_state = await state_manager()
    
    async with anyio.create_task_group() as tg:
        tg.start_soon(update_state)
        tg.start_soon(process_audio, state, state_send_stream)
        tg.start_soon(process_video, state, state_send_stream)
        tg.start_soon(llm_process, state, state_send_stream)

if __name__ == "__main__":
    anyio.run(main)