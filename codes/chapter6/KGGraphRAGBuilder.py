import os
import codecs

from openai.resources import api_key
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"


# 1、读入测试数据---------------------------------------------------------------
# 指定测试数据的目录路径
# 有2个测试文件，第1个存放《悟空传》的第1~4章，第2个存放第5~7章。
# 用同一个主题测试，因为LLM提取的实体类型形同，比较好处理。
directory_path = '/home/jean/dataset/test'

# 读入测试文件。
def read_txt_files(directory):
    # 存放结果的列表
    results = []
    # 遍历指定目录下的所有文件和文件夹
    for filename in os.listdir(directory):
        # 检查文件扩展名是否为.txt
        if filename.endswith(".txt"):
            # 构建完整的文件路径
            file_path = os.path.join(directory, filename)
            # 打开并读取文件内容
            with codecs.open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 将文件名和内容以列表形式添加到结果列表
            results.append([filename, content])
    
    return results


# 调用函数并打印结果
file_contents = read_txt_files(directory_path)
for file_name, content in file_contents:
    print("文件名:", file_name)

# 2、文本分块-------------------------------------------------------------------
# 对于大的文本文件，要从中提取实体关系，LLM的处理能力有限，必须分块处理。
# 另外GraphRAG向量搜索只需要返回相关的片段，也需要文本分块。
# 使用HanLP进行文本分块，本地运行，可以使用GPU，厂商独立，比较方便。
# 为保证提取的知识图谱之间有更好的连接质量，文本块之间要有适当的重叠，
# 文本块开头的重叠部分与末尾的截断部分最好是完整的句子。
# 所以文本块的大小和重叠部分的大小要根据当前的文本内容动态调整。
import hanlp

# 单任务模型，分词，token的计数是计算词，包括标点符号。
# LLM的工作都是基于token，先分词，再分块。
tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

# 划分段落。
def split_into_paragraphs(text):
    return text.split('\n')

# 判断token是否为句子结束符，视情况再增加。
def is_sentence_end(token):
    return token in ['。', '！', '？']

# 向后查找到句子结束符，用于动态调整chunk划分以保证chunk以完整的句子结束
def find_sentence_boundary_forward(tokens, chunk_size):
    end = len(tokens)  # 默认的end值设置为tokens的长度
    for i in range(chunk_size, len(tokens)):  # 从chunk_size开始向后查找
        if is_sentence_end(tokens[i]):
            end = i + 1  # 包含句尾符号
            break
    return end  

# 从位置start开始向前寻找上一句的句子结束符，以保证分块重叠的部分从一个完整的句子开始。
def find_sentence_boundary_backward(tokens, start):
    for i in range(start - 1, -1, -1):
        if is_sentence_end(tokens[i]):
            return i + 1  # 包含句尾符号
    return 0  # 找不到
  

# 文本分块，文本块的参考大小为chunk_size，文本块之间重叠部分的参考大小为overlap。
# 为了保证文本块之间重叠的部分及文本块末尾截断的部分都是完整的句子，
# 文本块的大小和重叠部分的大小都是根据当前文本块的内容动态调整的，是浮动的值。
def chunk_text(text, chunk_size=300, overlap=50):
    if chunk_size <= overlap:  # 参数检查
        raise ValueError("chunk_size must be greater than overlap.")
    # 先划分为段落，段落保存了语义上的信息，整个段落去处理。  
    paragraphs = split_into_paragraphs(text)
    chunks = []
    buffer = []
    # 逐个段落处理
    i = 0
    while i < len(paragraphs):
        # 注满buffer，直到大于chunk_szie，整个段落读入，段落保存了语义上的信息。
        while len(buffer) < chunk_size and i < len(paragraphs):
            tokens = tokenizer(paragraphs[i])
            buffer.extend(tokens)
            i += 1
        # 当前buffer分块
        while len(buffer) >= chunk_size:
            # 保证从完整的句子处截断。
            end = find_sentence_boundary_forward(buffer, chunk_size)
            chunk = buffer[:end]
            chunks.append(chunk)  # 保留token的状态以便后面计数
            # 保证重叠的部分从完整的句子开始。
            start_next = find_sentence_boundary_backward(buffer, end - overlap)
            if start_next==0:  # 找不到了上一句的句子结束符，调整重叠范围再找一次。
                start_next = find_sentence_boundary_backward(buffer, end-1)
            if start_next==0:  # 真的找不到，放弃块首的完整句子重叠。
                start_next = end - overlap
            buffer=buffer[start_next:]

        
    if buffer:  # 如果缓冲区还有剩余的token
        # 检查一下剩余部分是否已经包含在最后一个分块之中，它只是留作块间重叠。
        last_chunk = chunks[len(chunks)-1]
        rest = ''.join(buffer)
        temp = ''.join(last_chunk[len(last_chunk)-len(rest):])
        if temp!=rest:   # 如果不是留作重叠，则是最后的一个分块。
            chunks.append(buffer)
    
    return chunks


# 使用自定义函数进行分块
for file_content in file_contents:
    print("文件名:", file_content[0])
    chunks = chunk_text(file_content[1], chunk_size=500, overlap=50)
    file_content.append(chunks)
    
# 打印分块结果
for file_content in file_contents:
    print(f"File: {file_content[0]} Chunks: {len(file_content[2])}")
    for i, chunk in enumerate(file_content[2]):
        print(f"Chunk {i+1}: {len(chunk)} tokens.")


print(''.join(file_contents[0][2][0]))
print(''.join(file_contents[0][2][1]))
print(''.join(file_contents[0][2][6]))

# 3、在Neo4j中创建文档与Chunk的图结构-------------------------------------------
# https://python.langchain.com/v0.2/api_reference/community/graphs/langchain_community.graphs.neo4j_graph.Neo4jGraph.html#langchain_community.graphs.neo4j_graph.Neo4jGraph.add_graph_documents
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
import hashlib
import logging
from typing import List

# os.environ["NEO4J_URI"] = "bolt+ssc://localhost:7687"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

graph = Neo4jGraph(refresh_schema=False)


# 创建Document结点，与Chunk之间按属性名fileName匹配。
def create_Document(graph,type,uri,file_name, domain):
    query = """
    MERGE(d:`__Document2__` {fileName :$file_name}) SET d.type=$type,
          d.uri=$uri, d.domain=$domain
    RETURN d;
    """
    doc = graph.query(query,{"file_name":file_name,"type":type,"uri":uri,"domain":domain})
    return doc
  
# 创建Document结点
for file_content in file_contents:
    doc = create_Document(graph,"local",directory_path,file_content[0],"《悟空传》")


#创建Chunk结点并建立Chunk之间及与Document之间的关系
#这个程序直接从Neo4j KG Builder拷贝引用，为了增加tokens属性稍作修改。
#https://github.com/neo4j-labs/llm-graph-builder/blob/main/backend/src/make_relationships.py
def create_relation_between_chunks(graph, file_name, chunks: List)->list:
    logging.info("creating FIRST_CHUNK and NEXT_CHUNK relationships between chunks")
    current_chunk_id = ""
    lst_chunks_including_hash = []
    batch_data = []
    relationships = []
    offset=0
    for i, chunk in enumerate(chunks):
        page_content = ''.join(chunk)
        page_content_sha1 = hashlib.sha1(page_content.encode()) # chunk.page_content.encode()
        previous_chunk_id = current_chunk_id
        current_chunk_id = page_content_sha1.hexdigest()
        position = i + 1 
        if i>0:
            last_page_content = ''.join(chunks[i-1])
            offset += len(last_page_content)  # chunks[i-1].page_content
        if i == 0:
            firstChunk = True
        else:
            firstChunk = False  
        metadata = {"position": position,"length": len(page_content), "content_offset":offset, "tokens":len(chunk)}
        chunk_document = Document(
            page_content=page_content, metadata=metadata
        )
        
        chunk_data = {
            "id": current_chunk_id,
            "pg_content": chunk_document.page_content,
            "position": position,
            "length": chunk_document.metadata["length"],
            "f_name": file_name,
            "previous_id" : previous_chunk_id,
            "content_offset" : offset,
            "tokens" : len(chunk)
        }
        
        batch_data.append(chunk_data)
        
        lst_chunks_including_hash.append({'chunk_id': current_chunk_id, 'chunk_doc': chunk_document})
        
        # create relationships between chunks
        if firstChunk:
            relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
        else:
            relationships.append({
                "type": "NEXT_CHUNK",
                "previous_chunk_id": previous_chunk_id,  # ID of previous chunk
                "current_chunk_id": current_chunk_id
            })
          
    query_to_create_chunk_and_PART_OF_relation = """
        UNWIND $batch_data AS data
        MERGE (c:`__Chunk2__` {id: data.id})
        SET c.text = data.pg_content, c.position = data.position, c.length = data.length, c.fileName=data.f_name, 
            c.content_offset=data.content_offset, c.tokens=data.tokens
        WITH data, c
        MATCH (d:`__Document2__` {fileName: data.f_name})
        MERGE (c)-[:PART_OF]->(d)
    """
    graph.query(query_to_create_chunk_and_PART_OF_relation, params={"batch_data": batch_data})
    
    query_to_create_FIRST_relation = """ 
        UNWIND $relationships AS relationship
        MATCH (d:`__Document2__` {fileName: $f_name})
        MATCH (c:`__Chunk2__` {id: relationship.chunk_id})
        FOREACH(r IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
                MERGE (d)-[:FIRST_CHUNK]->(c))
        """
    graph.query(query_to_create_FIRST_relation, params={"f_name": file_name, "relationships": relationships})   
    
    query_to_create_NEXT_CHUNK_relation = """ 
        UNWIND $relationships AS relationship
        MATCH (c:`__Chunk2__` {id: relationship.current_chunk_id})
        WITH c, relationship
        MATCH (pc:`__Chunk2__` {id: relationship.previous_chunk_id})
        FOREACH(r IN CASE WHEN relationship.type = 'NEXT_CHUNK' THEN [1] ELSE [] END |
                MERGE (c)<-[:NEXT_CHUNK]-(pc))
        """
    graph.query(query_to_create_NEXT_CHUNK_relation, params={"relationships": relationships})   
    
    return lst_chunks_including_hash


#创建Chunk结点并建立Chunk之间及与Document之间的关系
for file_content in file_contents:
    file_name = file_content[0]
    chunks = file_content[2]
    result = create_relation_between_chunks(graph, file_name , chunks)
    file_content.append(result)

# 4、用LLM在每个块文本中提取实体关系--------------------------------------------

import os
import sys
sys.path.append("/home/jean/python")

from LangChainHelper import loadLLM, loadEmbedding
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

# llm = loadLLM("OpenAI")
# llm = loadLLM("Baidu")
# llm = loadLLM("Xunfei")
# llm = loadLLM("Tengxun")
# llm = loadLLM("Ali")
# llm = loadLLM("Ollama")
llm = loadLLM("Siliconflow")

system_template="""
-目标- 
给定相关的文本文档和实体类型列表，从文本中识别出这些类型的所有实体以及所识别实体之间的所有关系。 
-步骤- 
1.识别所有实体。对于每个已识别的实体，提取以下信息： 
-entity_name：实体名称，大写 
-entity_type：以下类型之一：[{entity_types}]
-entity_description：对实体属性和活动的综合描述 
将每个实体格式化为("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>
2.从步骤1中识别的实体中，识别彼此*明显相关*的所有实体配对(source_entity, target_entity)。 
对于每对相关实体，提取以下信息： 
-source_entity：源实体的名称，确保源实体在步骤1中所标识的实体之中
-target_entity：目标实体的名称，确保目标实体在步骤1中所标识的实体之中
-relationship_type：以下类型之一：[{relationship_types}]，当不能归类为上述列表中前面的类型时，归类为最后的一类“其它”
-relationship_description：解释为什么你认为源实体和目标实体是相互关联的 
-relationship_strength：一个数字评分，表示源实体和目标实体之间关系的强度 
将每个关系格式化为("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>) 
3.实体和关系的所有属性用中文输出，步骤1和2中识别的所有实体和关系输出为一个列表。使用{record_delimiter}作为列表分隔符。 
4.完成后，输出{completion_delimiter}

###################### 
-示例- 
###################### 
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"workmate"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"workmate"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"workmate"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"workmate"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"study"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"leaded by"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"operate"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}9){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"contact"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"leads"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"leads"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"controled by"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
#############################
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template="""
-真实数据- 
###################### 
实体类型：{entity_types} 
关系类型：{relationship_types}
文本：{input_text} 
###################### 
输出：
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, MessagesPlaceholder("chat_history"), human_message_prompt]
)

chain = chat_prompt | llm


tuple_delimiter = " : "
record_delimiter = "\n"
completion_delimiter = "\n\n"

entity_types = ["人物","妖怪","位置"]
relationship_types=["师徒", "师兄弟", "对抗", "对话", "态度", "故事地点", "其它"]
chat_history = []

import time
t0 = time.time()
for file_content in file_contents:
    results = []
    for chunk in file_content[2]:
        t1 = time.time()
        input_text = ''.join(chunk)
        answer = chain.invoke({
            "chat_history": chat_history,
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "tuple_delimiter": tuple_delimiter,
            "record_delimiter": record_delimiter,
            "completion_delimiter": completion_delimiter,
            "input_text": input_text
        })
        t2 = time.time()
        results.append(answer.content)
        print(input_text)
        print("\n")
        print(answer.content)
        print("块耗时：",t2-t1,"秒")
        print("\n")
        
    print("文件耗时：",t2-t0,"秒")
    print("\n\n")
    file_content.append(results)


# 提取的实体关系写入Neo4j-------------------------------------------------------
# 自己写代码由 answer.content生成一个GraphDocument对象
# 每个GraphDocument对象里增加一个metadata属性chunk_id，以便与前面建立的Chunk结点关联
import re
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

# 将每个块提取的实体关系文本转换为LangChain的GraphDocument对象
def convert_to_graph_document(chunk_id, input_text, result):
    # 提取节点和关系
    node_pattern = re.compile(r'\("entity" : "(.+?)" : "(.+?)" : "(.+?)"\)')
    relationship_pattern = re.compile(r'\("relationship" : "(.+?)" : "(.+?)" : "(.+?)" : "(.+?)" : (.+?)\)')

    nodes = {}
    relationships = []

    # 解析节点
    for match in node_pattern.findall(result):
        node_id, node_type, description = match
        if node_id not in nodes:
            nodes[node_id] = Node(id=node_id, type=node_type, properties={'description': description})

    # 解析并处理关系
    for match in relationship_pattern.findall(result):
        source_id, target_id, type, description, weight = match
        # 确保source节点存在
        if source_id not in nodes:
            nodes[source_id] = Node(id=source_id, type="未知", properties={'description': 'No additional data'})
        # 确保target节点存在
        if target_id not in nodes:
            nodes[target_id] = Node(id=target_id, type="未知", properties={'description': 'No additional data'})
        relationships.append(Relationship(source=nodes[source_id], target=nodes[target_id], type=type,
            properties={"description":description, "weight":float(weight)}))

    # 创建图对象
    graph_document = GraphDocument(
        nodes=list(nodes.values()),
        relationships=relationships,
        # page_content不能为空。
        source=Document(page_content=input_text, metadata={"chunk_id": chunk_id})
    )
    return graph_document


# 构造所有文档所有Chunk的GraphDocument对象
for file_content in file_contents:
    chunks = file_content[3]
    results = file_content[4]
    
    graph_documents = []
    for chunk, result in zip(chunks, results):
        graph_document =  convert_to_graph_document(chunk["chunk_id"] ,chunk["chunk_doc"].page_content, result)
        graph_documents.append(graph_document)
        # print(chunk)
        # print(result)
        # print(graph_document)
        # print("\n\n")
    file_content.append(graph_documents)
    

# 实体关系图写入Neo4j，此时每个Chunk是作为Documet结点创建的
# 后面再根据chunk_id把这个Document结点与相应的Chunk结点合并
for file_content in file_contents:
    # 删除没有识别出实体关系的空的图对象
    graph_documents = []
    for graph_document in file_content[5]:
        if len(graph_document.nodes)>0 or len(graph_document.relationships)>0:
            graph_documents.append(graph_document)
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )

# 合并Chunk结点与add_graph_documents()创建的相应Document结点，
# 迁移所有的实体关系到Chunk结点，并删除相应的Document结点。
# 完成Document->Chunk->Entity的结构。
def merge_relationship_between_chunk_and_entites(graph: Neo4jGraph, graph_documents_chunk_chunk_Id : list):
    batch_data = []
    logging.info("Create MENTIONS relationship between chunks and entities")
    for graph_doc_chunk_id in graph_documents_chunk_chunk_Id:
        query_data={
            'chunk_id': graph_doc_chunk_id,
        }
        batch_data.append(query_data)

    if batch_data:
        unwind_query = """
          UNWIND $batch_data AS data
          MATCH (c:`__Chunk2__` {id: data.chunk_id}), (d:Document{chunk_id:data.chunk_id})
          WITH c, d
          MATCH (d)-[r:MENTIONS]->(e)
          MERGE (c)-[newR:MENTIONS]->(e)
          ON CREATE SET newR += properties(r)
          DETACH DELETE d
                """
        graph.query(unwind_query, params={"batch_data": batch_data})


# 合并块结点与Document结点
for file_content in file_contents:
    graph_documents_chunk_chunk_Id=[]
    for chunk in file_content[3]:
        graph_documents_chunk_chunk_Id.append(chunk["chunk_id"])
    
    merge_relationship_between_chunk_and_entites(graph, graph_documents_chunk_chunk_Id)


# graph.add_graph_documents设置的base entity label 是__Entity__，改为__Entity2__
# 以与其它demo实例的实体区分
resc = graph.query("""
    match (e:`__Entity__`)-[r2]-(n:`__Chunk2__`)-[r]-(m:`__Document2__`) 
    remove e:`__Entity__`
    set e:`__Entity2__`

""")

# for file_content in file_contents:
#     file_content.pop()





# # 5、测试Embedding模型----------------------------------------------------------
# import os
# import sys
# sys.path.append("/home/ubuntu/Python")
# 
# from LangChainHelper import loadLLM, loadEmbedding
# import numpy as np
# 
# # 计算两个向量的余弦相似度
# def cosine_similarity(vector1, vector2):
#     # 计算向量的点积
#     dot_product = np.dot(vector1, vector2)
#     # 计算向量的范数（长度）
#     norm_vector1 = np.linalg.norm(vector1)
#     norm_vector2 = np.linalg.norm(vector2)
#     
#     # 检查是否存在零向量，避免除以零
#     if norm_vector1 == 0 or norm_vector2 == 0:
#         raise ValueError("输入向量之一是零向量，无法计算余弦相似度。")
#     
#     # 计算余弦相似度
#     cosine_sim = dot_product / (norm_vector1 * norm_vector2)
#     return cosine_sim
# 
# 
# # 加载BAAI/BGE-M3,消耗GPU 3.7G显存
# embeddings = loadEmbedding("BAAI")
# 
# # 长文本Embedding
# print(len(file_contents[0][1]))
# single_vector = embeddings.embed_query(file_contents[0][1])
# temp = file_contents[0][1][100:4100]
# single_vector2 = embeddings.embed_query(file_contents[0][1][100:4100])
# print(cosine_similarity(single_vector,single_vector2))
# 
# single_vector3 = embeddings.embed_query(file_contents[0][1][100:2000]+file_contents[0][1][2100:4100])
# print(cosine_similarity(single_vector,single_vector3))
# 
# 
# # 短文本Embedding
# tests=["沙僧","沙和尚","猪","猪八戒","孙悟空","悟空"]
# results = embeddings.embed_documents(tests)
# 
# print(cosine_similarity(results[0],results[1]))
# print(cosine_similarity(results[2],results[3]))
# print(cosine_similarity(results[4],results[5]))
# 
# 6、实体合并-------------------------------------------------------------------
import os
import sys
sys.path.append("/home/jean/python")

from LangChainHelper import loadLLM, loadEmbedding
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from graphdatascience import GraphDataScience

# os.environ["NEO4J_URI"] = "bolt+ssc://localhost:7687"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

graph = Neo4jGraph(refresh_schema=False)


# llm = loadLLM("OpenAI")
# llm = loadLLM("Baidu")
# llm = loadLLM("Xunfei")
# llm = loadLLM("Tengxun")
# llm = loadLLM("Ali")
# llm = loadLLM("Ollama")
llm = loadLLM("Siliconflow")

embeddings = loadEmbedding("BAAI")

# 用['id', 'description']来计算实体结点的Embedding。
vector = Neo4jVector.from_existing_graph(
    embeddings,
    node_label='__Entity2__',
    text_node_properties=['id', 'description'],
    embedding_node_property='embedding',
    index_name="vector2"
)


# GDS连接Neo4j
gds = GraphDataScience(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

# 用K近邻算法查找embedding相似值在阈值以内的近邻
# 建立所有实体在内存投影的子图，GDS算法都要通过内存投影运行
# G代表了子图的投影
G, result = gds.graph.project(
    "entities",                   #  Graph name
    "__Entity2__",                 #  Node projection
    "*",                          #  Relationship projection
    nodeProperties=["embedding"]  #  Configuration parameters
)

# 根据前面对Embedding模型的测试设置相似性阈值
similarity_threshold = 0.80 # 这是根据 OpenAI text-embedding-3-small 调整的阈值


# 用KNN算法找出Embedding相似的实体，建立SIMILAR连接
gds.knn.mutate(
  G,
  nodeProperties=['embedding'],
  mutateRelationshipType= 'SIMILAR',
  mutateProperty= 'score',
  similarityCutoff=similarity_threshold
)

# # 弱连接组件算法（不分方向），从新识别的SIMILAR关系中识别相识的社区，社区编号存放在结点的wcc属性
gds.wcc.write(
    G,
    writeProperty="wcc",
    relationshipTypes=["SIMILAR"]
)

# # 为了截图演示，再执行一次写入Neo4j磁盘存储
# gds.knn.write(
#   G,
#   nodeProperties=['embedding'],
#   writeRelationshipType= 'SIMILAR',
#   writeProperty= 'score',
#   similarityCutoff=similarity_threshold
# )
# 

# 找出潜在的相同实体
word_edit_distance = 5
potential_duplicate_candidates = graph.query(
    """MATCH (e:`__Entity2__`)
    WHERE size(e.id) > 1 // longer than 2 characters
    WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
    WHERE count > 1
    UNWIND nodes AS node
    // Add text distance
    WITH distinct
      [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id] AS intermediate_results
    WHERE size(intermediate_results) > 1
    WITH collect(intermediate_results) AS results
    // combine groups together if they share elements
    UNWIND range(0, size(results)-1, 1) as index
    WITH results, index, results[index] as result
    WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
            CASE WHEN index <> index2 AND
                size(apoc.coll.intersection(acc, results[index2])) > 0
                THEN apoc.coll.union(acc, results[index2])
                ELSE acc
            END
    )) as combinedResult
    WITH distinct(combinedResult) as combinedResult
    // extra filtering
    WITH collect(combinedResult) as allCombinedResults
    UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
    WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
    WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
        WHERE x <> combinedResultIndex
        AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
    )
    RETURN combinedResult
    """, params={'distance': word_edit_distance})


potential_duplicate_candidates[:5]

# # 看看LLM是否支持with_structured_output()
# if hasattr(llm, 'with_structured_output'):
#     print("This model supports with_structured_output.")
# else:
#     print("This model does not support with_structured_output.")


# 由LLM来最终决定哪些实体该合并
system_template = """
你是一名数据处理助理。您的任务是识别列表中的重复实体，并决定应合并哪些实体。 
这些实体在格式或内容上可能略有不同，但本质上指的是同一个实体。运用你的分析技能来确定重复的实体。 
以下是识别重复实体的规则： 
1.语义上差异较小的实体应被视为重复。 
2.格式不同但内容相同的实体应被视为重复。 
3.引用同一现实世界对象或概念的实体，即使描述不同，也应被视为重复。 
4.如果它指的是不同的数字、日期或产品型号，请不要合并实体。
输出格式：
1.将要合并的实体输出为Python列表的格式，输出时保持它们输入时的原文。
2.如果有多组可以合并的实体，每组输出为一个单独的列表，每组分开输出为一行。
3.如果没有要合并的实体，就输出一个空的列表。
4.只输出列表即可，不需要其它的说明。
5.不要输出嵌套的列表，只输出列表。
###################### 
-示例- 
###################### 
Example 1:
['Star Ocean The Second Story R', 'Star Ocean: The Second Story R', 'Star Ocean: A Research Journey']
#############
Output:
['Star Ocean The Second Story R', 'Star Ocean: The Second Story R']
#############################
Example 2:
['Sony', 'Sony Inc', 'Google', 'Google Inc', 'OpenAI']
#############
Output:
['Sony', 'Sony Inc']
['Google', 'Google Inc']
#############################
Example 3:
['December 16, 2023', 'December 2, 2023', 'December 23, 2023', 'December 26, 2023']
Output:
[]
#############################
"""
user_template = """
以下是要处理的实体列表： 
{entities} 
请识别重复的实体，提供可以合并的实体列表。
输出：
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(user_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, MessagesPlaceholder("chat_history"), human_message_prompt]
)

chain = chat_prompt | llm

# 调用LLM得到可以合并实体的列表
merged_entities=[]
for canditates in potential_duplicate_candidates:
    chat_history = []
    answer = chain.invoke({
        "chat_history": chat_history,
        "entities": canditates
        })
    merged_entities.append(answer.content)
    print(answer.content)

# 合并实体----------------------------------------------------------------------
import re
import ast

# 将每个实体列表文本转换为Python List
# 返回的是二级列表
def convert_to_list(result):
    list_pattern = re.compile(r'\[.*?\]')
    entity_lists = []

    # 解析实体列表
    for match in list_pattern.findall(result):
        # 使用 ast.literal_eval 将字符串解析为实际的Python列表
        try:
            entity_list = ast.literal_eval(match)
            entity_lists.append(entity_list)
        except Exception as e:
            print(f"Error parsing {match}: {e}")
    
    return entity_lists

# 最终可合并的实体列表是一个二级列表，每组可合并的实体一个列表。
results = []
for canditates in merged_entities:
    # 将返回的二级列表展平为一级列表
    temp = convert_to_list(canditates)
    for entities in temp:
        if (len(entities)>0):
            results.append(entities)
  
print(results)

# 合并实体
graph.query("""
UNWIND $data AS candidates
CALL {
  WITH candidates
  MATCH (e:__Entity2__) WHERE e.id IN candidates
  RETURN collect(e) AS nodes
}
CALL apoc.refactor.mergeNodes(nodes, {properties: {
    `.*`: 'discard'
}})
YIELD node
RETURN count(*)
""", params={"data": results})


# 处理完毕，删除内存中的子图投影
G.drop()

# 7、社区发现-------------------------------------------------------------------
# 1）Leiden算法-----------------------------------------------------------------
# # 建立子图投影
# G, result = gds.graph.project(
#     "communities",  #  Graph name
#     "__Entity2__",  #  Node projection
#     {
#         "_ALL_": {
#             "type": "*",
#             "orientation": "UNDIRECTED",
#             "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
#         }
#     },
# )
# 
# # 等效的Cypher投影语句：
# # https://neo4j.com/docs/graph-data-science/2.6/management-ops/graph-creation/graph-project/
# # Cypher语句中的MAP不等同于Python的字典，key不需要用引号括起，值才需要。
# # CALL gds.graph.project(
# #   'communities',                  // 图的名称
# #   '__Entity2__',                 // 节点投影
# #   {
# #   _ALL_: {             // 关系类型的标识符，表示所有类型关系的统一配置
# #       type: '*',                  // 匹配所有类型的关系
# #       orientation: 'UNDIRECTED',  // 将关系视为无向关系
# #       properties: {               // 定义关系属性的处理方式
# #         weight: {
# #           property: '*',          // 匹配所有属性
# #           aggregation: 'COUNT'    // 计数聚合；计算有多少条边满足条件
# #         }
# #       }
# #     }
# #   }
# # );
# 
# 
# wcc = gds.wcc.stats(G)
# print(f"Component count: {wcc['componentCount']}")
# print(f"Component distribution: {wcc['componentDistribution']}")
# 
# 
# gds.leiden.write(
#     G,
#     writeProperty="communities",
#     includeIntermediateCommunities=True,
#     relationshipWeightProperty="weight",
# )
# 
# graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community2__) REQUIRE c.id IS UNIQUE;")
# 
# graph.query("""
# MATCH (e:`__Entity2__`)
# UNWIND range(0, size(e.communities) - 1 , 1) AS index
# CALL {
#   WITH e, index
#   WITH e, index
#   WHERE index = 0
#   MERGE (c:`__Community2__` {id: toString(index) + '-' + toString(e.communities[index])})
#   ON CREATE SET c.level = index
#   MERGE (e)-[:IN_COMMUNITY]->(c)
#   RETURN count(*) AS count_0
# }
# CALL {
#   WITH e, index
#   WITH e, index
#   WHERE index > 0
#   MERGE (current:`__Community2__` {id: toString(index) + '-' + toString(e.communities[index])})
#   ON CREATE SET current.level = index
#   MERGE (previous:`__Community2__` {id: toString(index - 1) + '-' + toString(e.communities[index - 1])})
#   ON CREATE SET previous.level = index - 1
#   MERGE (previous)-[:IN_COMMUNITY]->(current)
#   RETURN count(*) AS count_1
# }
# RETURN count(*)
# """)
# 
# # 处理完毕，删除内存中的子图投影
# G.drop()
# 
# 2、SLLPA算法------------------------------------------------------------------
# 建立子图投影
G, result = gds.graph.project(
    "communities",  #  Graph name
    "__Entity2__",  #  Node projection
    {
        "_ALL_": {
            "type": "*",
            "orientation": "UNDIRECTED",
            "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
        }
    },
)

# 找到调用sllpa算法的名字
algorithms = gds.list()
slpas =  algorithms[algorithms['description'].str.contains("Propagation", case=False, na=False)]
print(slpas)

# 调用sllpa算法	
gds.sllpa.write(
    G,
    maxIterations=100,
    minAssociationStrength = 0.1,
)

graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community2__) REQUIRE c.id IS UNIQUE;")

graph.query("""
MATCH (e:`__Entity2__`)
UNWIND range(0, size(e.communityIds) - 1 , 1) AS index
CALL {
  WITH e, index
  MERGE (c:`__Community2__` {id: '0-'+toString(e.communityIds[index])})
  ON CREATE SET c.level = 0
  MERGE (e)-[:IN_COMMUNITY]->(c)
  RETURN count(*) AS count_0
}
RETURN count(*)
""")

# 处理完毕，删除内存中的子图投影
G.drop()

# ------------------------------------------------------------------------------
# 为社区增加权重community_rank
graph.query("""
MATCH (c:`__Community2__`)<-[:IN_COMMUNITY*]-(:`__Entity2__`)<-[:MENTIONS]-(d:`__Chunk2__`)
WITH c, count(distinct d) AS rank
SET c.community_rank = rank;
""")

# 检索社区所包含的结点与边的信息
community_info = graph.query("""
MATCH (c:`__Community2__`)<-[:IN_COMMUNITY*]-(e:__Entity2__)
WHERE c.level IN [0]
WITH c, collect(e ) AS nodes
WHERE size(nodes) > 1
CALL apoc.path.subgraphAll(nodes[0], {
	whitelistNodes:nodes
})
YIELD relationships
RETURN c.id AS communityId,
       [n in nodes | {id: n.id, description: n.description, type: [el in labels(n) WHERE el <> '__Entity2__'][0]}] AS nodes,
       [r in relationships | {start: startNode(r).id, type: type(r), end: endNode(r).id, description: r.description}] AS rels
""")

community_info


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

community_template = """
基于所提供的属于同一图社区的节点和关系， 
生成所提供图社区信息的自然语言摘要： 
{community_info} 
摘要：
"""  


community_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "给定一个输入三元组，生成信息摘要。没有序言。",
        ),
        ("human", community_template),
    ]
)

community_chain = community_prompt | llm | StrOutputParser()

def prepare_string(data):
    nodes_str = "Nodes are:\n"
    for node in data['nodes']:
        node_id = node['id']
        node_type = node['type']
        if 'description' in node and node['description']:
            node_description = f", description: {node['description']}"
        else:
            node_description = ""
        nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

    rels_str = "Relationships are:\n"
    for rel in data['rels']:
        start = rel['start']
        end = rel['end']
        rel_type = rel['type']
        if 'description' in rel and rel['description']:
            description = f", description: {rel['description']}"
        else:
            description = ""
        rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

    return nodes_str + "\n" + rels_str

def process_community(community):
    stringify_info = prepare_string(community)
    summary = community_chain.invoke({'community_info': stringify_info})
    return {"community": community['communityId'], "summary": summary, "full_content":stringify_info}

# 执行社区摘要  
summaries = []
for info in community_info:
    result = process_community(info)
    summaries.append(result)
    
for summary in summaries:
    print(summary["community"])
    print(summary["summary"])

# Store summaries
graph.query("""
UNWIND $data AS row
MERGE (c:__Community2__ {id:row.community})
SET c.summary = row.summary, c.full_content = row.full_content
""", params={"data": summaries})

