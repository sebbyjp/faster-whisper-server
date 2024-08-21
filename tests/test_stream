from mbodied.agents import LanguageAgent
from rich.pretty import pprint
agent_test = LanguageAgent(model_src="openai", api_key="mbodi-demo-1", model_kwargs={"base_url": "http://localhost:3389/v1"})
response = ""
for chunk in agent_test.act_and_stream(instruction="Tell me about the weather in New York", model="astro-world"):
    response += chunk
    pprint(response)
    # break

# resp = agent_test.act(instruction="Tell me about the weather in New York", model="astro-world")

# from openai import OpenAI
# openai = OpenAI(api_key="mbodi-demo-1", base_url="http://localhost:3389/v1")
# with openai.chat.with_streaming_response.completions.create(messages=[
#   {"role": "system", "content": "You are a helpful assistant."},
#   {"role": "user", "content": "What is the weather in New York?"}
# ], model="astro-world", stream=True
# ) as response:
#     resp = ""
#     for chunk in response.iter_text():

#         pprint(chunk)

