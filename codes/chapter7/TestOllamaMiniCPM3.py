from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain_core.tools import tool

llm = ChatOllama(
    model="minicpm3-4b-fp16", # qwen2.5, qwen2.5:14b, minicpm3-4b-fp16
    temperature=0,
    base_url="http://localhost:11434/"
    # other params...
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是个有用的助手，可以回答人们的问题。",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input": "请介绍一下你自己。"
    }
)

@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.
    """
    return True


llm_with_tools = llm.bind_tools([validate_user])
result = llm_with_tools.invoke(
    "Could you validate user 123? They previously lived at "
    "123 Fake St in Boston MA and 234 Pretend Boulevard in "
    "Houston TX."
)
result.tool_calls
