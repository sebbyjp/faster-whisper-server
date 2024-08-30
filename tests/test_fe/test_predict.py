import asyncio
from collections.abc import AsyncGenerator
from typing import Literal

import pytest

PredInstrMode = Literal["predict", "repeat", "clear", "wait"]
NOT_A_COMPLETE_INSTRUCTION = "Not a complete instruction..."
DETERMINE_INSTRUCTION_PROMPT = "Determine if the following is a complete instruction:"

# Mock LanguageAgent for testing purposes
class MockLanguageAgent:
    async def act(self, instruction, model, extra_body) -> str:
        # Simulate the agent's behavior
        if "complete" in instruction.lower():
            return "Yes"
        return "No"

    def forget(self, everything) -> None:
        pass

weak_agent = MockLanguageAgent()

@pytest.mark.asyncio
async def predict_instruction(text: str, mode: PredInstrMode, last_instruction: str) -> AsyncGenerator[tuple[str, PredInstrMode], None]:
    if mode == "clear" or not text or not text.strip():
        yield "", "wait"
        return
    if mode == "repeat":
        yield last_instruction, mode
        return

    if text == NOT_A_COMPLETE_INSTRUCTION:
        print("Instruction is not a complete instruction.")
        yield NOT_A_COMPLETE_INSTRUCTION, "predict"
        return

    print(f"Text: {text}, Mode Predict: {mode}, Last instruction: {last_instruction}")

    weak_agent.forget(everything=True)
    full_instruction = DETERMINE_INSTRUCTION_PROMPT + "\n" + text
    if await weak_agent.act(instruction=full_instruction, model="astroworld", extra_body={"guided_choice": ["Yes", "No"]}) == "Yes":
        print(f"Instruction: is a complete instruction. Returning {text}")
        yield text, "repeat"
    else:
        print(f"Instruction: {text} is not a complete instruction.")
        yield NOT_A_COMPLETE_INSTRUCTION, "predict"

        print(await weak_agent.act(instruction="Why wasn't it a complete instruction?", model="astroworld"))
        yield NOT_A_COMPLETE_INSTRUCTION, "predict"

@pytest.mark.asyncio
async def test_predict_instruction() -> None:
    # Test case 1: Empty input
    result = [r async for r in predict_instruction("", "predict", "")]
    assert result == [("", "wait")], f"Expected [('', 'wait')], got {result}"

    # Test case 2: Repeat mode
    result = [r async for r in predict_instruction("test", "repeat", "last instruction")]
    assert result == [("last instruction", "repeat")], f"Expected [('last instruction', 'repeat')], got {result}"

    # Test case 3: Incomplete instruction
    result = [r async for r in predict_instruction("incomplete", "predict", "")]
    assert result == [(NOT_A_COMPLETE_INSTRUCTION, "predict"), (NOT_A_COMPLETE_INSTRUCTION, "predict")], f"Expected [(NOT_A_COMPLETE_INSTRUCTION, 'predict'), (NOT_A_COMPLETE_INSTRUCTION, 'predict')], got {result}"

    # Test case 4: Complete instruction
    result = [r async for r in predict_instruction("This is a complete instruction", "predict", "")]
    assert result == [("This is a complete instruction", "repeat")], f"Expected [('This is a complete instruction', 'repeat')], got {result}"

    print("All tests passed for predict_instruction!")

if __name__ == "__main__":
    asyncio.run(test_predict_instruction())
