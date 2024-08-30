from collections.abc import AsyncGenerator
from time import time
from typing import Literal

import pytest

ActMode = Literal["acting", "repeat", "clear", "wait"]

class MockLanguageAgent:
    def __init__(self) -> None:
        self.history_length = 0

    async def act_and_stream(self, instruction, model) -> AsyncGenerator[str, None]:
        words = instruction.split()
        for word in words:
            yield word + " "

    def history(self) -> list:
        return [None] * self.history_length

    def forget_after(self, n) -> None:
        self.history_length = n

@pytest.fixture
def agent() -> MockLanguageAgent:
    return MockLanguageAgent()

async def act(instruction: str, last_response: str, last_tps: str, mode: ActMode, agent) -> AsyncGenerator[tuple[str, str, ActMode], None]:
    print(f"Instruction: {instruction}, last response: {last_response}, mode: {mode}")

    if mode == "clear" or not instruction or not instruction.strip():
        yield "", last_tps, "wait"
        return
    if mode == "wait":
        yield "", last_tps, "wait"
        return
    if mode == "repeat":
        yield last_response, last_tps, "repeat"
        return
    if mode not in ["acting", "wait"]:
        raise ValueError(f"Invalid mode: {mode}")

    if len(agent.history()) > 10:
        agent.forget_after(2)

    tic = time()
    total_tokens = 0
    response = ""
    instruction = instruction + "\n Answer briefly and concisely."
    print(f"Following instruction: {instruction}.")
    last_response = ""
    async for text in agent.act_and_stream(instruction=instruction, model="astroworld"):
        total_tokens = len(response.split())
        elapsed_time = time() - tic
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        response += text
        print(f"Response: {response}")
        if response == last_response:
            yield response, f"TPS: {tokens_per_sec:.4f}", "repeat"
            return
        yield response, f"TPS: {tokens_per_sec:.4f}", "acting"

@pytest.mark.asyncio
async def test_act_empty_input(agent) -> None:
    result = [r async for r in act("", "", "", "clear", agent)]
    assert result == [("", "", "wait")], f"Expected [('', '', 'wait')], got {result}"

@pytest.mark.asyncio
async def test_act_wait_mode(agent) -> None:
    result = [r async for r in act("test", "", "", "wait", agent)]
    assert result == [("", "", "wait")], f"Expected [('', '', 'wait')], got {result}"

@pytest.mark.asyncio
async def test_act_repeat_mode(agent) -> None:
    result = [r async for r in act("test", "last response", "last_tps", "repeat", agent)]
    assert result == [("last response", "last_tps", "repeat")], f"Expected [('last response', 'last_tps', 'repeat')], got {result}"

@pytest.mark.asyncio
async def test_act_acting_mode(agent) -> None:
    results = [r async for r in act("This is a test instruction", "", "", "acting", agent)]
    assert len(results) > 0, "Expected at least one result"
    assert all(isinstance(r[0], str) and r[0] for r in results), "Expected non-empty string responses"
    assert all(r[1].startswith("TPS:") for r in results), "Expected TPS information"
    assert all(r[2] == "acting" for r in results), "Expected 'acting' mode for all results"
    assert "This is a test instruction Answer briefly and concisely." in results[-1][0], "Final response should contain the full instruction"

@pytest.mark.asyncio
async def test_act_invalid_mode(agent) -> None:
    with pytest.raises(ValueError):
        [r async for r in act("test", "", "", "invalid", agent)]

if __name__ == "__main__":
    pytest.main([__file__])
