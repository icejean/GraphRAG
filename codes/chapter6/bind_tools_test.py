import os
import sys
from keys import keys

sys.path.append("/home/jean/python")
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"

from LangChainHelper import loadLLM, loadEmbedding
from langchain_core.tools import tool

# 定义一个工具函数
@tool
def greeting(name: str) ->str : 
    '''向朋友致欢迎语'''
    return f"你好啊, {name}"
 

# chat_vendors = ["OpenAI","Baidu","Xunfei","Tengxun","Ali","Ollama","Siliconflow"]
chat_vendors = ["Siliconflow"]

for vendor in chat_vendors:
    try:
        print("\nVendor: "+vendor+"\n")
        llm = loadLLM(vendor)
        llm_with_tool = llm.bind_tools([greeting])
        resp = llm_with_tool.invoke("你好，我是Jean，问候一下。")
        print(resp)
    except Exception as e:
        print(e)

# ------------------------------------------------------------------------------
from langchain_core.tools import tool

# 定义一个工具函数
@tool
def greeting(name: str) ->str : 
    '''向朋友致欢迎语'''
    return f"你好啊, {name}"
 

from langchain_ollama import ChatOllama
llm = ChatOllama(
      model="MFDoom/deepseek-r1-tool-calling:14b",  # qwen2.5:14b, MFDoom/deepseek-r1-tool-calling:14b
      temperature=0,
      base_url="http://localhost:11434/"
      # other params...
)
llm_with_tool = llm.bind_tools([greeting])
# MiniCPM3目前不支持工具调用。
# ollama._types.ResponseError: deepseek-r1:14b does not support tools
resp = llm_with_tool.invoke("你好，我是Jean，问候一下。")
print(resp)

