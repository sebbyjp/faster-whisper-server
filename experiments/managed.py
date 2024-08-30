import asyncio
import json

import aioredis


class StateManager:
    def __init__(self, redis_url):
        self.redis = None
        self.redis_url = redis_url
        self.stream_key = "state_stream"

    async def connect(self):
        self.redis = await aioredis.from_url(self.redis_url)

    async def update_state(self, update):
        if not self.redis:
            await self.connect()
        await self.redis.xadd(self.stream_key, update)

    async def get_state(self):
        if not self.redis:
            await self.connect()
        # Get the latest entry from the stream
        latest = await self.redis.xrevrange(self.stream_key, count=1)
        if latest:
            return json.loads(latest[0][1][b'data'])
        return {}

    async def stream_state(self):
        if not self.redis:
            await self.connect()
        latest_id = '0-0'
        while True:
            updates = await self.redis.xread([self.stream_key], latest_id=latest_id, count=1, block=0)
            for stream, entries in updates:
                for entry_id, entry in entries:
                    latest_id = entry_id
                    yield json.loads(entry[b'data'])

state_manager = StateManager("redis://localhost")

# Usage in async context
async def some_async_function():
    await state_manager.update_state({'data': json.dumps({'audio': {'new_data': 'value'}})})
    current_state = await state_manager.get_state()
    # Use current_state...

# To continuously listen for updates
async def listen_for_updates():
    async for update in state_manager.stream_state():
        print("New state update:", update)

from fastapi import FastAPI
import asyncio

app = FastAPI()
state_manager = StateManager("redis://localhost")

@app.on_event("startup")
async def startup_event():
    await state_manager.connect()

@app.get("/state")
async def get_state():
    return await state_manager.get_state()

@app.post("/update")
async def update_state(update: dict):
    await state_manager.update_state({'data': json.dumps(update)})
    return {"status": "updated"}

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        async for update in state_manager.stream_state():
            await websocket.send_json(update)
    except WebSocketDisconnect:
        pass