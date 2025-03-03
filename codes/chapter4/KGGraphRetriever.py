# 1、设置环境-------------------------------------------------------------------
import os
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"

from neo4j import GraphDatabase, Result
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from typing import Dict, Any
import pandas as pd
from openai.resources import api_key



# LLM以多段的形式回答问题。
response_type: str = "多个段落"

# 设置Neo4j的运行参数
NEO4J_URI="bolt://winhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="password"
# Neo4j向量索引的名字
index_name = "vector"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# 执行Cypher查询
def db_query(cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
    """Executes a Cypher statement and returns a DataFrame"""
    return driver.execute_query(
        cypher, parameters_=params, result_transformer_=Result.to_df
    )

# 为社区结点设置权重以便检索时排序
db_query("""
MATCH (n:`__Community__`)<-[:IN_COMMUNITY]-()<-[:MENTIONS]-(c)
WITH n, count(distinct c) AS chunkCount
SET n.weight = chunkCount
""")

os.environ["OPENAI_API_KEY"] = api_key.key
llm =ChatOpenAI(model="gpt-4o")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", cache_folder="/home/ubuntu/.cache/huggingface/hub/"
)


# 2、局部查询-------------------------------------------------------------------
topChunks = 3
topCommunities = 3
topOutsideRels = 10
topInsideRels = 10
topEntities = 10

lc_retrieval_query = """
WITH collect(node) as nodes
// Entity - Text Unit Mapping
WITH
collect {
    UNWIND nodes as n
    MATCH (n)<-[:MENTIONS]-(c:__Chunk__)
    WITH distinct c, count(distinct n) as freq
    RETURN {id:c.id, text: c.text} AS chunkText
    ORDER BY freq DESC
    LIMIT $topChunks
} AS text_mapping,
// Entity - Report Mapping
collect {
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
    WITH distinct c, c.community_rank as rank, c.weight AS weight
    RETURN c.summary 
    ORDER BY rank, weight DESC
    LIMIT $topCommunities
} AS report_mapping,
// Outside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r]-(m:__Entity__) 
    WHERE NOT m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.weight DESC 
    LIMIT $topOutsideRels
} as outsideRels,
// Inside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r]-(m:__Entity__) 
    WHERE m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.weight DESC 
    LIMIT $topInsideRels
} as insideRels,
// Entities description
collect {
    UNWIND nodes as n
    RETURN n.description AS descriptionText
} as entities
// We don't have covariates or claims here
RETURN {Chunks: text_mapping, Reports: report_mapping, 
       Relationships: outsideRels + insideRels, 
       Entities: entities} AS text, 1.0 AS score, {} AS metadata
"""

LC_SYSTEM_PROMPT="""
---角色--- 
您是一个有用的助手，请根据用户输入的上下文，综合上下文中多个分析报告的数据，来回答问题，并遵守回答要求。

---任务描述--- 
总结来自多个不同分析报告的数据，生成要求长度和格式的回复，以回答用户的问题。 

---回答要求---
- 你要严格根据分析报告的内容回答，禁止根据常识和已知信息回答问题。
- 对于不知道的问题，直接回答“不知道”。
- 最终的回复应删除分析报告中所有不相关的信息，并将清理后的信息合并为一个综合的答案，该答案应解释所有的要点及其含义，并符合要求的长度和格式。 
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。 
- 回复应保留之前包含在分析报告中的所有数据引用，但不要提及各个分析报告在分析过程中的作用。 
- 如果回复引用了Entities、Reports及Relationships类型分析报告中的数据，则用它们的顺序号作为ID。
- 如果回复引用了Chunks类型分析报告中的数据，则用原始数据的id作为ID。 
- **不要在一个引用中列出超过5个引用记录的ID**，相反，列出前5个最相关的引用记录ID。 
- 不要包括没有提供支持证据的信息。
例如： 
#############################
“X是Y公司的所有者，他也是X公司的首席执行官，他受到许多违规行为指控，其中的一些已经涉嫌违法。” 

{{'data': {{'Entities':[3], 'Reports':[2, 6], 'Relationships':[12, 13, 15, 16, 64], 'Chunks':['d0509111239ae77ef1c630458a9eca372fb204d6','74509e55ff43bc35d42129e9198cd3c897f56ecb'] }} }}
#############################
---回复的长度和格式--- 
- {response_type}
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。  
- 在回复的最后才输出数据引用的情况，单独作为一段。
输出引用数据的格式：
{{'data': {{'Entities':[逗号分隔的顺序号列表], 'Reports':[逗号分隔的顺序号列表], 'Relationships':[逗号分隔的顺序号列表], 'Chunks':[逗号分隔的id列表] }} }}
例如：
{{'data': {{'Entities':[3], 'Reports':[2, 6], 'Relationships':[12, 13, 15, 16, 64], 'Chunks':['d0509111239ae77ef1c630458a9eca372fb204d6','74509e55ff43bc35d42129e9198cd3c897f56ecb'] }} }}

"""

# 局部检索器
def local_retriever(query: str, response_type: str = response_type) -> str:
    # 局部检索的提示词
    lc_prompt = ChatPromptTemplate.from_messages(
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
                
                {report_data}
                

                用户的问题是：
                {question}
                """,
            ),
        ]
    )
    # 局部检索的chain
    lc_chain = lc_prompt | llm | StrOutputParser()
    # 局部检索的Neo4j向量存储与索引
    lc_vector = Neo4jVector.from_existing_index(
        embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=index_name,
        retrieval_query=lc_retrieval_query,
    )
    # 先进行向量相似性搜索
    docs = lc_vector.similarity_search(
        query,
        k=topEntities,
        params={
            "topChunks": topChunks,
            "topCommunities": topCommunities,
            "topOutsideRels": topOutsideRels,
            "topInsideRels": topInsideRels,
        },
    )
    
    print(docs[0].page_content)
    
    # 向量相似性搜索的结果注入提示词并提交给LLM
    lc_response = lc_chain.invoke(
        {
            "report_data": docs[0].page_content,
            "question": query,
            "response_type": response_type,
        }
    )
    # 返回LLM的答复
    return lc_response

answer = local_retriever("孙悟空跟女妖之间有什么故事？")
print(answer)


answer2 = local_retriever("红孩儿是谁家的孩子？")
print(answer2)


# 3、全局查询-------------------------------------------------------------------
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


# 全局检索器
def global_retriever(query: str, level: int, response_type: str = response_type) -> str:
    # MAP阶段生成中间结果的prompt与chain
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
        RETURN c.summary AS output
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
        print(intermediate_response)
    # 再用LLM从每个社区摘要生成的中间结果生成最终的答复
    final_response = reduce_chain.invoke(
        {
            "report_data": intermediate_results,
            "question": query,
            "response_type": response_type,
        }
    )
    # 返回LLM最终的答复
    return final_response


answer3 = global_retriever("唐僧的取经团队对路上遇到的妖怪采取什么样的态度？", 0)
print(answer3)


answer4 = global_retriever("孙悟空对唐僧是一种什么样的心态？", 0)
print(answer4)
