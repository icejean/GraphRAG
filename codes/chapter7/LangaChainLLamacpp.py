# 测试面壁用LlamaCpp 本地运行MiniCPM3 4B模型

import os
from openai.resources import api_key
from keys import keys

os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"

# Path to your model weights
# Minicpm3-ggml-model-Q4_K_M.gguf, minicpm3-4b-fp16.gguf
local_model = "/home/demo/dataset/gguf/minicpm3-4b-fp16.gguf" 

import multiprocessing

from langchain_community.chat_models import ChatLlamaCpp

llm = ChatLlamaCpp(
    temperature=0.1,
    model_path=local_model,
    n_ctx=8192, # 1024, 2048, 4096, 8192, 我的内存最大上下文窗口就到8K。
    n_gpu_layers=63,
    n_batch=16,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    max_tokens=1024,
    n_threads=multiprocessing.cpu_count() - 1,
    repeat_penalty=1.5,
    top_p=0.5,
    verbose=True,
)

messages = [
    (
        "system",
        "你是一个有用的助手。",
    ),
    ("human", "秋天去中国的哪里旅游比较合适？"),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)

from langchain_core.prompts import PromptTemplate

template = """
问题: {question}
答案: 
"""
prompt = PromptTemplate.from_template(template)
chain = prompt | llm

question = "秋天去中国的哪里旅游比较合适？"
ai_msg2= chain.invoke({"question": question})
print(ai_msg2.content)



from langchain_core.tools import tool
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(enum=["celsius", "fahrenheit"])


@tool("get_current_weather", args_schema=WeatherInput)
def get_weather(location: str, unit: str):
    """Get the current weather in a given location"""
    return f"Now the weather in {location} is 22 {unit}"

@tool
def get_population(location: str, unit: str):
    """Get the current weather in a given location"""
    return f"Now the population in {location} is 10000 {unit}"


# https://python.langchain.com/docs/integrations/chat/llamacpp/
# LangChain文档例子中用tool_choice参数强制模型调用工具，而不是根据需要调用工具，
# 有点取巧，并不符合Agent实际应用场景中根据需要调用工具的要求。
# 这是Llamacpp的问题。
# https://github.com/abetlen/llama-cpp-python/issues/1338
# https://github.com/abetlen/llama-cpp-python/discussions/1615
# https://github.com/abetlen/llama-cpp-python/issues/1784

llm_with_tools = llm.bind_tools(
    tools=[get_weather],
    tool_choice={"type": "function", "function": {"name": "get_current_weather"}},
)

ai_msg3 = llm_with_tools.invoke(
    "what is the weather like in HCMC in celsius",
)

print(ai_msg3.tool_calls)


# 测试按需自动选择工具，输出直接回答问题了，没有根据需要决定是否需要调用工具。
llm_with_tools2 = llm.bind_tools(tools=[get_weather, get_population])
ai_msg4 = llm_with_tools2.invoke(
    "what is the weather like in HCMC in celsius",
)
print(ai_msg4)

