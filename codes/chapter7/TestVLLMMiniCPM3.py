# 1、Chat测试-----------------------------------------------------
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openbmb/MiniCPM3-4B",
    openai_api_key="your key",
    openai_api_base="http://localhost:4000/v1",
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to Italian."
    ),
    HumanMessage(
        content="Translate the following sentence from English to Italian: I love programming."
    ),
]
llm.invoke(messages)

# 2、Chain测试----------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)

# 3、工具调用测试---------------------------------------------------
from langchain_core.tools import tool

# 定义一个工具函数
@tool
def greeting(name: str) ->str : 
    '''向朋友致欢迎语'''
    return f"你好啊, {name}"
 
llm_with_tool = llm.bind_tools([greeting])
resp = llm_with_tool.invoke("你好，我是Jean，问候一下。")
print(resp)
