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

# 全局检索器MAP阶段系统提示词
MAP_SYSTEM_PROMPT = """
---角色--- 
你是一位有用的助手，可以回答有关所提供表格中数据的问题。 

---任务描述--- 
- 生成一个回答用户问题所需的要点列表，总结输入数据表格中的所有相关信息。 
- 你应该使用下面数据表格中提供的数据作为生成回复的主要上下文。
- 你要严格根据提供的数据表格来回答问题，当提供的数据表格中没有足够的信息时才运用自己的知识。
- 如果你不知道答案，或者提供的数据表格中没有足够的信息来提供答案，就说不知道。不要编造任何答案。
- 不要包括没有提供支持证据的信息。
- 数据支持的要点应列出相关的数据引用作为参考，并列出产生该要点社区的communityId。
- **不要在一个引用中列出超过5个引用记录的ID**。相反，列出前5个最相关引用记录的顺序号作为ID。

---回答要求---
回复中的每个要点都应包含以下元素： 
- 描述：对该要点的综合描述。 
- 重要性评分：0-100之间的整数分数，表示该要点在回答用户问题时的重要性。“不知道”类型的回答应该得0分。 

---回复的格式--- 
回复应采用JSON格式，如下所示： 
{{ 
"points": [ 
{{"description": "Description of point 1 {{'nodes': [nodes list seperated by comma], 'relationships':[relationships list seperated by comma], 'communityId': communityId form context data}}", "score": score_value}}, 
{{"description": "Description of point 2 {{'nodes': [nodes list seperated by comma], 'relationships':[relationships list seperated by comma], 'communityId': communityId form context data}}", "score": score_value}}, 
] 
}}

例如： 

{{"points": [
{{"description": "X是Y公司的所有者，他也是X公司的首席执行官。 {{'nodes': [1,3], 'relationships':[2,4,6,8,9], 'communityId':'0-0'}}", "score": 80}}, 
{{"description": "X受到许多不法行为指控。 {{'nodes': [1,3], 'relationships':[12,14,16,18,19], 'communityId':'0-0'}}", "score": 90}}
] 
}}

"""

# 全局检索器REDUCE阶段系统提示词
REDUCE_SYSTEM_PROMPT = """
---角色--- 
你是一个有用的助手，请根据用户输入的上下文，综合上下文中多个要点列表的数据，来回答问题，并遵守回答要求。

---任务描述--- 
总结来自多个不同要点列表的数据，生成要求长度和格式的回复，以回答用户的问题。 

---回答要求---
- 你要严格根据要点列表的内容回答，禁止根据常识和已知信息回答问题。
- 对于不知道的信息，直接回答“不知道”。
- 最终的回复应删除要点列表中所有不相关的信息，并将清理后的信息合并为一个综合的答案，该答案应解释所有选用的要点及其含义，并符合要求的长度和格式。 
- 根据要求的长度和格式，把回复划分为适当的章节和段落。 
- 回复应保留之前包含在要点列表中的要点引用，并且包含引用要点来源社区原始的communityId，但不要提及各个要点在分析过程中的作用。 
- **不要在一个引用中列出超过5个要点引用的ID**，相反，列出前5个最相关要点引用的顺序号作为ID。 
- 不要包括没有提供支持证据的信息。

例如： 

“X是Y公司的所有者，他也是X公司的首席执行官{{'points':[(1,'0-0'),(3,'0-0')]}}，
受到许多不法行为指控{{'points':[(2,'0-0'), (3,'0-0'), (6,'0-1'), (9,'0-1'), (10,'0-3')]}}。” 
其中1、2、3、6、9、10表示相关要点引用的顺序号，'0-0'、'0-1'、'0-3'是要点来源的communityId。 

---回复的长度和格式--- 
- {response_type}
- 根据要求的长度和格式，把回复划分为适当的章节和段落。  
- 输出要点引用的格式：
{{'points': [逗号分隔的要点元组]}}
每个要点元组的格式如下：
(要点顺序号, 来源社区的communityId)
例如：
{{'points':[(1,'0-0'),(3,'0-0')]}}
{{'points':[(2,'0-0'), (3,'0-0'), (6,'0-1'), (9,'0-1'), (10,'0-3')]}}
- 要点引用的说明放在引用之后，不要单独作为一段。

例如： 

“X是Y公司的所有者，他也是X公司的首席执行官{{'points':[(1,'0-0'),(3,'0-0')]}}，
受到许多不法行为指控{{'points':[(2,'0-0'), (3,'0-0'), (6,'0-1'), (9,'0-1'), (10,'0-3')]}}。” 
其中1、2、3、6、9、10表示相关要点引用的顺序号，'0-0'、'0-1'、'0-3'是要点来源的communityId。

"""

# 4、全局检索器 ----------------------------------------------------------------

from langchain_community.graphs import Neo4jGraph
from langchain_core.tools import tool
from tqdm import tqdm

level =0

# 全局检索器
@tool
def global_retriever(query: str) -> str:
    """回答有关网络小说《悟空传》的全书的全局性问题。"""
    # 上面这段工具功能描述是必须的，否则调用工具会出错。
    
    # 检索MAP阶段生成中间结果的prompt与chain
    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                MAP_SYSTEM_PROMPT,
            ),
            (
                "human",
                """
                ---数据表格--- 
                {context_data}
                
                
                用户的问题是：
                {question}
                """,
            ),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    # 连接Neo4j
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        refresh_schema=False,
    )
    # 检索指定层级的社区
    community_data = graph.query(
        """
        MATCH (c:__Community__)
        WHERE c.level = $level
        RETURN {communityId:c.id, full_content:c.full_content} AS output
        """,
        params={"level": level},
    )
    # 用LLM从每个检索到的社区摘要生成中间结果
    intermediate_results = []
    for community in tqdm(community_data, desc="Processing communities"):
        intermediate_response = map_chain.invoke(
            {"question": query, "context_data": community["output"]}
        )
        intermediate_results.append(intermediate_response)
        # 输出看一下
        # print(intermediate_response)
        
    # 返回一个ToolMessage，包含每个社区对问题总结的要点，直接返回字典列表即可。
    return intermediate_results



# 5、局部检索器-----------------------------------------------------------------
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "wu_kong_zhuan_local_retriever",
    "检索网络小说《悟空传》中各章节的人物与故事情节。", # 工具的功能描述是必须的
)

# 工具列表有2个工具，局部检索器和全局检索器，LLM会根据工具的描述决定用哪一个。
tools = [retriever_tool, global_retriever]


# 6、自定义Agent----------------------------------------------------------------

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

### Edges ----------------------------------------------------------------------

# 分流边
# 这个自定义的LangGraph边对工具调用的结果进行分流处理。
# 如果是全局检索，转到reduce结点生成回复。
# 如果是局部检索并且检索结果与问题相关，转到generate结点生成回复。
# 如果是局部检索并且检索结果与问题不相关，转到rewrite结点重构问题。
# 局部检索的结果是否与问题相关，提交给LLM去判断。
# 画流程图时要用到Literal。
def grade_documents(state) -> Literal["generate", "rewrite", "reduce"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    messages = state["messages"]
    
    # 倒数第2条消息是LLM发出工具调用请求的AIMessage。
    retrieve_message = messages[-2]
    
    # print(retrieve_message)
    
    # 如果是全局查询直接转去reduce结点。
    # if retrieve_message.additional_kwargs["tool_calls"][0]["function"]["name"]== 'global_retriever':
    if retrieve_message.tool_calls[0]["name"]== 'global_retriever':
      
        print("---Global retrieve---")
        return "reduce"

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
    # 最后一条消息是检索器返回的结果。
    last_message = messages[-1]
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content
    
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})
    # LLM会给出检索结果与问题是否相关的判断, yes或no
    score = scored_result.content
    # 保险起见要转为小写！！！
    # if score.lower() == "yes":
    if "yes" in score.lower():      
        print("---DECISION: DOCS RELEVANT---")
        print(score)
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


### Nodes ----------------------------------------------------------------------

# Agent结点
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
    # 这里为Agent绑定前面定义的工具集：局部检索和全局检索。
    model = model.bind_tools(tools)
    try:
        response = model.invoke(messages)
    except Exception as e:
        print(str(e))
        response = str(e)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# 重构问题结点
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

# 局部检索回复生成结点
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
    # 检索结果在最后一条消息中。
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
  
# 全局查询回复生成结点。
def reduce(state):
    """
    Generate answer for global retrieve

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---REDUCE---")
    messages = state["messages"]
    
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content
    # 检索结果在最后一条消息中。
    last_message = messages[-1]
    docs = last_message.content

    # Reduce阶段生成最终结果的prompt与chain
    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                REDUCE_SYSTEM_PROMPT,
            ),
            (
                "human",
                """
                ---分析报告--- 
                {report_data}


                用户的问题是：
                {question}
                """,
            ),
        ]
    )
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    # 再用LLM从每个社区摘要生成的中间结果生成最终的答复
    response = reduce_chain.invoke(
        {
            "report_data": docs,
            "question": question,
            "response_type": response_type,
        }
    )
    # 明确返回一条AIMessage。
    return {"messages": [AIMessage(content = response)]}

    
# Graph ------------------------------------------------------------------------
# Agent的工作流图定义。
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 在内存中管理对话历史
memory = MemorySaver()

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode(tools)
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
# 增加一个全局查询的reduce结点
workflow.add_node(
    "reduce", reduce
)

# 定义结点之间的连接
# 从agent结点开始
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,          # tools_condition()的输出是"tools"或END
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",  # 转到retrieve结点，执行局部检索或全局检索
        END: END,             # 直接结束
    },
)

# 检索结点执行结束后调边grade_documents，决定流转到哪个结点: generate、rewrite、reduce。
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
# 如果是局部查询生成，直接结束
workflow.add_edge("generate", END)
# 如果是重构问题，转到agent结点重新开始。
workflow.add_edge("rewrite", "agent")
# 增加一条全局查询到结束的边
workflow.add_edge("reduce", END)


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

ask_agent("《悟空传》书中的主要人物有哪些？",graph,config)
print(get_answer(config))
ask_agent("孙悟空跟女妖之间有什么故事？",graph,config)
print(get_answer(config))
ask_agent("他最后的选择是什么？",graph,config)
print(get_answer(config))
ask_agent("其他人怎样看女妖？",graph,config)
print(get_answer(config))


chat_history = memory.get(config)["channel_values"]["messages"]

# 画流程图
png_data = graph.get_graph().draw_mermaid_png()
# 指定将要保存的文件名
file_path = '/home/ubuntu/images/workflow2.png'
# 打开一个文件用于写入二进制数据
with open(file_path, 'wb') as f:
    f.write(png_data)

