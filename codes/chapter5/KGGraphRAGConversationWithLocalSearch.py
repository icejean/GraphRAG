# 1、设置运行环境 --------------------------------------------------------------
import os
from openai.resources import api_key
from keys import keys
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable

from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
os.environ["OPENAI_API_KEY"] = api_key.key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = keys.huggingface_key
# 使用LangSmith记录LLM调用的日志，可在在https://smith.langchain.com上查看详情
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = keys.langchain_api_key
# os.environ["LANGSMITH_PROJECT"] = "My Project Name" # Optional: "default" is used if not set

llm = ChatOpenAI(model="gpt-4o")   # gpt-4o-mini, gpt-4o
# https://huggingface.co/docs/transformers/installation#offline-mode
# 本地运行Embedding模型，设置为OFFLINE模式，不需要连接网络。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", cache_folder="/home/ubuntu/.cache/huggingface/hub/"
)

# 设置Neo4j的运行参数
# NEO4J_URI="bolt+ssc://localhost:7687"
NEO4J_URI="bolt://winhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="password"
# Neo4j向量索引的名字
index_name = "vector"

# 2、实例化局部检索的Neo4j向量存储与索引 ---------------------------------------
# 定义局部查询检索器参数
topChunks = 3
topCommunities = 3
topOutsideRels = 10
topInsideRels = 10
topEntities = 10

# Neo4jVector.as_retriever()不支持动态的传入Cypher查询语句的参数，所以改成用参数动态的生成Cypher查询语句。
# https://python.langchain.com/v0.2/api_reference/community/vectorstores/langchain_community.vectorstores.neo4j_vector.Neo4jVector.html#langchain_community.vectorstores.neo4j_vector.Neo4jVector.as_retriever
lc_retrieval_query = f"""
WITH collect(node) as nodes
// Entity - Text Unit Mapping
WITH
collect {{
    UNWIND nodes as n
    MATCH (n)<-[:MENTIONS]-(c:__Chunk__)
    WITH distinct c, count(distinct n) as freq
    RETURN {{id:c.id, text: c.text}} AS chunkText
    ORDER BY freq DESC
    LIMIT {topChunks}
}} AS text_mapping,
// Entity - Report Mapping
collect {{
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
    WITH distinct c, c.community_rank as rank, c.weight AS weight
    RETURN c.summary 
    ORDER BY rank, weight DESC
    LIMIT {topCommunities}
}} AS report_mapping,
// Outside Relationships 
collect {{
    UNWIND nodes as n
    MATCH (n)-[r]-(m:__Entity__) 
    WHERE NOT m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.weight DESC 
    LIMIT {topOutsideRels}
}} as outsideRels,
// Inside Relationships 
collect {{
    UNWIND nodes as n
    MATCH (n)-[r]-(m:__Entity__) 
    WHERE m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.weight DESC 
    LIMIT {topInsideRels}
}} as insideRels,
// Entities description
collect {{
    UNWIND nodes as n
    RETURN n.description AS descriptionText
}} as entities
// We don't have covariates or claims here
RETURN {{Chunks: text_mapping, Reports: report_mapping, 
       Relationships: outsideRels + insideRels, 
       Entities: entities}} AS text, 1.0 AS score, {{}} AS metadata
"""

# 加载向量索引存储
lc_vector = Neo4jVector.from_existing_index(
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name=index_name,
    retrieval_query=lc_retrieval_query,
)
# 定义向量检索器
retriever = lc_vector.as_retriever(search_kwargs ={"k":topEntities})

# LLM以多段的形式回答问题。
response_type = "多个段落"


# 3、系统提示词 ----------------------------------------------------------------
#局部查询的系统提示词
LC_SYSTEM_PROMPT="""
---角色--- 
您是一个有用的助手，请根据用户输入的上下文，综合上下文中多个分析报告的数据，来回答问题，并遵守回答要求。

---任务描述--- 
总结来自多个不同分析报告的数据，生成要求长度和格式的回复，以回答用户的问题。 

---回答要求---
- 你要严格根据分析报告的内容回答，禁止根据常识和已知信息回答问题。
- 对于不知道的问题，直接回答“不知道”。
- 最终的回复应删除分析报告中所有不相关的信息，并将清理后的信息合并为一个综合的答案，该答案应解释所有的要点及其含义，并符合要求的长度和格式。 
- 根据要求的长度和格式，把回复划分为适当的章节和段落。 
- 回复应保留之前包含在分析报告中的所有数据引用，但不要提及各个分析报告在分析过程中的作用。 
- 如果回复引用了Entities、Reports及Relationships类型分析报告中的数据，则用它们的顺序号作为ID。
- 如果回复引用了Chunks类型分析报告中的数据，则用原始数据的id作为ID。 
- **不要在一个引用中列出超过5个引用记录的ID**，相反，列出前5个最相关的引用记录ID。 
- 不要包括没有提供支持证据的信息。

例如： 

“X是Y公司的所有者，他也是X公司的首席执行官，他受到许多违规行为指控，其中的一些已经涉嫌违法。” 

{{'data': {{'Entities':[3], 'Reports':[2, 6], 'Relationships':[12, 13, 15, 16, 64], 'Chunks':['d0509111239ae77ef1c630458a9eca372fb204d6','74509e55ff43bc35d42129e9198cd3c897f56ecb'] }} }}

---回复的长度和格式--- 
- {response_type}
- 根据要求的长度和格式，把回复划分为适当的章节和段落。  
- 在回复的最后才输出数据引用的情况，单独作为一段。
输出引用数据的格式：
{{'data': {{'Entities':[逗号分隔的顺序号列表], 'Reports':[逗号分隔的顺序号列表], 'Relationships':[逗号分隔的顺序号列表], 'Chunks':[逗号分隔的id列表] }} }}
例如：
{{'data': {{'Entities':[3], 'Reports':[2, 6], 'Relationships':[12, 13, 15, 16, 64], 'Chunks':['d0509111239ae77ef1c630458a9eca372fb204d6','74509e55ff43bc35d42129e9198cd3c897f56ecb'] }} }}

"""


# 3、建立可以处理对话历史上下文的 GraphRAG 查询Chain----------------------------
# 定义上下文检索器，它的作用是根据对话历史和当前问题，构造一个独立的问题，
# 然后在向量数据库中检索与这个问题相关的语料。
# 上下文检索器的系统提示词
contextualize_q_system_prompt ="""
给定一组聊天记录和最新的用户问题 ，
该问题可能会引用聊天记录中的上下文， 
据此构造一个不需要聊天记录也可以理解的独立问题， 
不要回答它。
如果需要，就重新构造出上述的独立问题，否则按原样返回原来的问题。
"""

#上下文检索器的提示词
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# 定义上下文检索器
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 局部查询的提示词
lc_prompt_with_history = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            LC_SYSTEM_PROMPT,
        ),
        MessagesPlaceholder("chat_history"),        
        (
            "human",
            """
            ---分析报告--- 
            请注意，下面提供的分析报告按**重要性降序排列**。
                
            {context}
                

            用户的问题是：
            {input}
            """,
        ),
    ]
)

# 定义局部查询的问答链
question_answer_chain_with_history = create_stuff_documents_chain(llm, lc_prompt_with_history)
# 定义支持上下文的局部查询问答链
rag_chain_with_history = create_retrieval_chain(history_aware_retriever, question_answer_chain_with_history)

# 对话记录
chat_history = []

# 定义可用LangSmith跟踪的局部查询函数
# LLM调用会被记录在https://smith.langchain.com上，可以登录查看详情。
@traceable
def local_retriever(question, chat_history):
    ai_msg = rag_chain_with_history.invoke({"input": question, "response_type":response_type, "chat_history": chat_history})
    return ai_msg["answer"]

# 4、测试-----------------------------------------------------------------------  
question1 = "孙悟空跟女妖之间有什么故事？"
answer1 = local_retriever(question1, chat_history)
print(answer1)
chat_history.extend(
    [
        HumanMessage(content=question1),
        AIMessage(content=answer1),
    ]
)
# 这个问题需要知道上下文，他是谁。
question2 = "他最后的选择是什么？"
answer2 = local_retriever(question2, chat_history)
print(answer2)
chat_history.extend(
    [
        HumanMessage(content=question2),
        AIMessage(content=answer2),
    ]
)
# 这个问题需要知道上下文，她是谁。
question3 = "她为什么变得丑陋？"
answer3 = local_retriever(question3, chat_history)
print(answer3)
chat_history.extend(
    [
        HumanMessage(content=question3),
        AIMessage(content=answer3),
    ]
)


# 5、用Agent实现局部查询--------------------------------------------------------
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 定义Agent可调用的工具列表
# 第一个工具是局部检索器
# retriever见前面的定义
# 定义向量检索器
# retriever = lc_vector.as_retriever(search_kwargs ={"k":topEntities})

tool = create_retriever_tool(
    retriever,
    "wu_kong_zhuan_local_retriever",
    "检索网络小说《悟空传》中各章节的人物与故事情节。",
)
# 工具列表中暂时只有一个工具局部检索器。
tools = [tool]

# 记录对话历史
memory = MemorySaver()
# 创建一个Agent
agent_executor = create_react_agent(llm, tools, checkpointer=memory)
# 会话的 session id
config = {"configurable": {"thread_id": "abc126"}}

def ask_agent(query,agent,config):
    for s in agent.stream(
        {"messages": [HumanMessage(content=query)]}, config=config
    ):
        print(s)
        print("----")

# 第1个问题，Agent决定不调用工具。
query1 = "你好，想问一些问题。"
ask_agent(query1,agent_executor,config)
chat_history = memory.get(config)["channel_values"]["messages"]
answer1 = chat_history[-1].content
print(answer1)

# 第2个问题，Agent决定调用工具中的局部检索器。
query2 = "孙悟空跟女妖之间有什么故事？"
ask_agent(query2,agent_executor,config)
chat_history = memory.get(config)["channel_values"]["messages"]
answer2 = chat_history[-1].content
print(answer2)

# 第3个问题，Agent决定根据上下文直接回答。
query3 = "他最后的选择是什么？"
ask_agent(query3,agent_executor,config)
chat_history = memory.get(config)["channel_values"]["messages"]
answer3 = chat_history[-1].content
print(answer3)

# 第4个问题，Agent决定根据上下文直接回答。
query4 = "她为什么变得丑陋？"
ask_agent(query4,agent_executor,config)
chat_history = memory.get(config)["channel_values"]["messages"]
answer4 = chat_history[-1].content
print(answer4)


# 6、自定义Agent----------------------------------------------------------------
# 要先运行前面第1、2节实例化 retriever变量。
# Retriever --------------------------------------------------------------------
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "wu_kong_zhuan_local_retriever",
    "检索网络小说《悟空传》中各章节的人物与故事情节。",
)
# 工具列表中暂时只有一个工具局部检索器，要为全局检索器也定义一个工具。
tools = [retriever_tool]

# Agent state ------------------------------------------------------------------
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 调试时如果要查看每个节点输入输出的状态，可以用这个函数插入打印的语句
def my_add_messages(left,right):
    print("\nLeft:\n")
    print(left)
    print("\nRight\n")
    print(right)
    return add_messages(left,right)
    
class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    # messages: Annotated[Sequence[BaseMessage], my_add_messages]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
# Nodes and Edges --------------------------------------------------------------
from typing import Annotated, Literal, Sequence, TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import tools_condition

### Edges

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # LLM
    model = llm

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    
    # Chain
    chain = prompt | llm

    messages = state["messages"]
    last_message = messages[-1]

    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content
    
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.content

    if "yes" in score.lower():
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"

### Nodes

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = llm

    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Rewriter
    model = llm

    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content
    
    last_message = messages[-1]

    docs = last_message.content

    # 局部查询的提示词
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                LC_SYSTEM_PROMPT,
            ),
            (
                "human",
                """
                ---分析报告--- 
                请注意，下面提供的分析报告按**重要性降序排列**。
                
                {context}
                

                用户的问题是：
                {question}
                """,
            ),
        ]
    )
    
    # LLM
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question, "response_type":response_type})
    
    # 这里有个Bug，response是String，generate节点返回的消息会自动判定为HumanMessage，其实是AIMessage。
    # 明确返回一条AIMessage。
    return {"messages": [AIMessage(content = response)]}
  

# Graph ------------------------------------------------------------------------
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 管理对话历史
memory = MemorySaver()

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `retrieve` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile(checkpointer=memory)

# Run --------------------------------------------------------------------------
import pprint

# 限制rewrite的次数，以免陷入无限的循环
config = {"configurable": {"thread_id": "226", "recursion_limit":5}}

def ask_agent(query,agent, config):
    inputs = {"messages": [("user", query)]}
    for output in agent.stream(inputs, config=config):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

def get_answer(config):
    chat_history = memory.get(config)["channel_values"]["messages"]
    answer = chat_history[-1].content
    return answer

ask_agent("你好，想问一些问题。",graph,config)
print(get_answer(config))
ask_agent("孙悟空跟女妖之间有什么故事？",graph,config)
print(get_answer(config))
ask_agent("他最后的选择是什么？",graph,config)
print(get_answer(config))
ask_agent("她为什么变得丑陋？",graph,config)
print(get_answer(config))


chat_history = memory.get(config)["channel_values"]["messages"]

# 画流程图
png_data = graph.get_graph().draw_mermaid_png()
# 指定将要保存的文件名
file_path = '/home/ubuntu/images/workflow.png'
# 打开一个文件用于写入二进制数据
with open(file_path, 'wb') as f:
    f.write(png_data)
