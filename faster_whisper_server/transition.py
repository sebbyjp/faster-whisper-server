from threading import Lock
from lager import log
lock = Lock()
state = {
    "pred_instr_mode": "predict",
    "transcription": "",
    "instruction": "",
    "response": "",
    "spoken": "",
    "uncommitted": "",
    "speak_mode": "wait",
    "act_mode": "wait",
}

def get_state():
    with lock:
        return state.copy()

def update_state(new_state):
    with lock:
        state.update(new_state)


def process_state(state: dict, raw_transcription: str, new_instruction: str, raw_response) -> dict:
    log.info(f"PROCESS: Current state: {state}, new transcription: {raw_transcription}, new instruction: {new_instruction}, raw response: {raw_response}")
    update_state({
        "pred_instr_mode": "predict",
        "transcription": raw_transcription,
        "instruction": new_instruction,
        "response": raw_response,
        "spoken": "",
    })
    return state, raw_transcription, new_instruction, raw_response

















    
    # overlap_spoken_uncommitted = spoken and uncommitted and find_overlap(spoken, uncommitted)
    # if overlap_spoken_uncoemmitted:
    #     console.print(f"Overlap: spoken uncommitted {overlap_spoken_uncommitted}", style="bold magenta on yellow")
    # uncommitted = uncommitted[len(overlap_spoken_uncommitted) :].strip()

    # # Update uncommitted and response
    # # Find overlap where response starts wi th end of uncommitted
    # overlap_uncommitted_response = spoken and uncommitted and find_overlap(uncommitted, response)
    # if overlap_uncommitted_response:
    #     console.print(f"Overlap: uncommitted response {overlap_uncommitted_response}", style="bold red on yellow")
    #     new_response = response[len(overlap_uncommitted_response) :].strip()
    #     uncommitted += " " + new_response
    # else:
    #     uncommitted += " " + response

    # uncommitted = uncommitted.removeprefix(find_overlap(spoken, uncommitted)).strip()
    # if raw_response:
    #     un

# def process_state(new_transcription: str, new_instruction: str, raw_response: str) -> dict:
#     state = get_state()
#     print(
#         f"PROCESS: Current state: {state}, new transcription: {new_transcription}, new instruction: {new_instruction}, raw response: {raw_response}"
#         f"uncommitted: {state.get('uncommitted', '')}"
#     )
#     spoken = state.get("spoken", "")
#     uncommitted = state["uncommitted"]
#     new_instruction = state["instruction"]
#     # response = state["response"]
#     # If any mode is "clear", reset all modes
#     if any(state.get(mode) == "clear" for mode in ["pred_instr_mode", "act_mode", "speak_mode"]):
#         update_state(
#             {
#                 "pred_instr_mode": "predict",
#                 "act_mode": "wait",
#                 "speak_mode": "wait",
#                 "transcription": "",
#                 "instruction": "",
#                 "uncommitted": "",
#             }
#         )
#         return (
#             {
#                 "pred_instr_mode": "predict",
#                 "act_mode": "wait",
#                 "speak_mode": "wait",
#                 "transcription": "",
#                 "instruction": "",
#                 "spoken": "",
#                 "uncommitted": "",
#             },
#             "",
#             "",
#             "",
#         )
#     spoken = state.get("spoken", "")
#     uncommitted = state.get("uncommitted", "")

#     # overlap_spoken_uncommitted = spoken and uncommitted and find_overlap(spoken, uncommitted)
#     # if overlap_spoken_uncoemmitted:
#     #     console.print(f"Overlap: spoken uncommitted {overlap_spoken_uncommitted}", style="bold magenta on yellow")
#     # uncommitted = uncommitted[len(overlap_spoken_uncommitted) :].strip()

#     # # Update uncommitted and response
#     # # Find overlap where response starts wi th end of uncommitted
#     # overlap_uncommitted_response = spoken and uncommitted and find_overlap(uncommitted, response)
#     # if overlap_uncommitted_response:
#     #     console.print(f"Overlap: uncommitted response {overlap_uncommitted_response}", style="bold red on yellow")
#     #     new_response = response[len(overlap_uncommitted_response) :].strip()
#     #     uncommitted += " " + new_response
#     # else:
#     #     uncommitted += " " + response

#     # uncommitted = uncommitted.removeprefix(find_overlap(spoken, uncommitted)).strip()
#     # if raw_response:
#     #     uncommitted += " " + raw_response.removeprefix(find_overlap(uncommitted, raw_response)).strip()
#     # yprint(f"Uncommitted: {uncommitted}")
#     # state["response"] = uncommitted
#     # update_state(state)
#     if state["speak_mode"] == "speaking":
#         return state, state["transcription"], state["instruction"], state["response"]
#         # Find overlap where uncommitted starts with end of spoken


#     # # if spoken is equal to response and uncommitted is empty, reset state
#     # if spoken == response and not uncommitted and (spoken and response):
#     #     update_state(
#     #         {
#     #             "pred_instr_mode": "predict",
#     #             "act_mode": "wait",
#     #             "speak_mode": "wait",
#     #             "transcription": "",
#     #             "instruction": "",
#     #             "response": "",
#     #             "spoken": "",
#     #             "audio_array": np.array([0], dtype=np.int16),
#     #             "uncommitted": "",
#     #         }
#     #     )
#     #     state = get_state()

#     # return state, state["transcription"], state["instruction"], state["response"]

#     # If no complete instruction yet
#     # Wait for more transcription before predicting
#     if not new_transcription: #or not new_transcription.strip() or new_transcription == "Not enough audio yet.":
#         # yprint(f"Waiting for more audio. Current transcription: {new_transcription}")

#         update_state(
#             {
#                 "pred_instr_mode": "wait",
#                 "act_mode": "wait",
#                 "speak_mode": "wait",
#                 "transcription": new_transcription,
#                 "instruction": new_instruction,
#                 "response": uncommitted,
#             }
#         )
#         return (
#             {
#                 "pred_instr_mode": "predict",
#                 "act_mode": "wait",
#                 "speak_mode": "wait",
#                 "transcription": new_transcription,
#                 "instruction": new_instruction,
#                 "response": uncommitted,
#             },
#             new_transcription,
#             new_instruction,
#             uncommitted,
#         )

#     # If we have an incomplete instruction
#     # Continue or start predicting
#     if new_transcription:
#         # yprint(f"Predicting instruction for: {new_transcription}")
#         update_state(
#             {
#                 "pred_instr_mode": "predict",
#                 "act_mode": "wait",
#                 "speak_mode": "wait",
#                 "transcription": new_transcription,
#                 "instruction": new_instruction,
#                 "response": uncommitted,
#             }
#         )
#         return (
#             {
#                 "pred_instr_mode": "predict",
#                 "act_mode": "wait",
#                 "speak_mode": "wait",
#                 "transcription": new_transcription,
#                 "instruction": new_transcription,
#                 "response": uncommitted,
#             },
#             new_transcription,
#             new_instruction,
#             raw_response,
#             uncommitted,
#         )

#     # If we have an instruction but haven't acted on it
#     # Repeat transcription and instruction
#     if new_instruction and state["act_mode"] == "wait":
#         # yprint(f"Acting on instruction: {new_instruction}")
#         update_state(
#             {
#                 "pred_instr_mode": "repeat",
#                 "act_mode": "acting",
#                 "speak_mode": "wait",
#                 "transcription": state["transcription"],
#                 "instruction": state["instruction"],
#                 "response": uncommitted,
#             }
#         )
#         return (
#             {
#                 "pred_instr_mode": "repeat",
#                 "act_mode": "acting",
#                 "speak_mode": "wait",
#                 "transcription": state["transcription"],
#                 "instruction": state["instruction"],
#             },
#             state["transcription"],
#             state["instruction"],
#             uncommitted,
#         )

#     # If we're acting and have a response, but haven't started speaking
#     if state["act_mode"] == "acting" and uncommitted and state["speak_mode"] in ("wait", "speaking"):
#         yprint(f"Speaking response: {uncommitted}")
#         update_state(
#             {
#                 "pred_instr_mode": "predict",
#                 "act_mode": "repeat",
#                 "speak_mode": "speaking",
#                 "transcription": get_state()["transcription"],
#                 "instruction": get_state()["instruction"],
#                 "response": uncommitted,
#             }
#         )
#         return (
#             {
#                 "pred_instr_mode": "predict",
#                 "act_mode": "repeat",
#                 "speak_mode": "speaking",
#                 "transcription": get_state()["transcription"],
#                 "instruction": get_state()["instruction"],
#                 "response": uncommitted,
#             },
#             state["transcription"],
#             state["instruction"],
#             uncommitted,
#         )
#     # If we've finished speaking
#     if spoken == raw_response and not uncommitted and (spoken and raw_response) and state["speak_mode"] == "finished":
#         gprint("Done speaking.")
#         update_state(
#             {
#                 "pred_instr_mode": "predict",
#                 "act_mode": "wait",
#                 "speak_mode": "wait",
#                 "transcription": "",
#                 "instruction": "",
#                 "response": "",
#                 "spoken": "",
#                 "audio_array": None,
#                 "uncommitted": "",
#             }
#         )
#         update_audio({"stream": None})
#         return state, new_transcription, new_instruction, uncommitted
#     # Default case: maintain current state
#     return state, state["transcription"], state["instruction"], state["response"]