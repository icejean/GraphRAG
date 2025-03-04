import os
from keys import keys

os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = keys.huggingface_key

import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "openbmb/MiniCPM3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", # cuda
    trust_remote_code=True,
    offload_folder="offload",
    offload_state_dict = True, 
    offload_buffers=True,
    torch_dtype=torch.float16,
)

pipe = pipeline("text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=256)

# 测试Huggingface pipeline模型是否可以运行。------------------------------------
messages = [
    {
        "role": "system",
        "content": "你是一个有用的助手，根据你的知识回答人们的问题。",
    },
    {"role": "user", "content": "秋天去中国的哪里旅游比较合适？"},
]
import time
response = pipe(messages, max_new_tokens=512)[0]['generated_text'][-1] # Print the assistant's response
print(response["content"])

# 测试 LangChain集成 -----------------------------------------------------------
from langchain_core.prompts import PromptTemplate

hf = HuggingFacePipeline(pipeline=pipe)

template = """
Question: {question}
Answer: 
"""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "秋天去中国的哪里旅游比较合适？"
response2 = chain.invoke({"question": question})
print(response["content"])


# 要加上tokenizer，否则会报错。
chat_model = ChatHuggingFace(llm=hf, tokenizer =tokenizer)


from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="你是个有用的助手。"),
        ("human", "{question}"),
    ]
)

output_parser = StrOutputParser()

chain = prompt | chat_model | output_parser
  
response = chain.invoke({"question": "秋天去中国的哪里旅游比较合适？"})
print(response)



# ------------------------------------------------------------------------------
from langchain_core.tools import tool

@tool
def greeting(name: str) ->str : 
    '''向朋友致欢迎语'''
    return f"你好啊, {name}"
 
# AttributeError: 'HuggingFacePipeline' object has no attribute 'bind_tools'
llm_with_tool = hf.bind_tools([greeting])
resp = llm_with_tool.invoke("你好，我是Jean")
print(resp)

