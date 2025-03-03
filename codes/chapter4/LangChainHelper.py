import os
from openai.resources import api_key
from keys import keys

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.chat_models import ChatSparkLLM
from langchain_community.embeddings import SparkLLMTextEmbeddings
from langchain_community.chat_models import ChatHunyuan
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
os.environ["OPENAI_API_KEY"] = api_key.key
os.environ["QIANFAN_AK"] = keys.API_KEY
os.environ["QIANFAN_SK"] = keys.SECRET_KEY
os.environ["HUNYUAN_SECRET_ID"] = keys.tx_id
os.environ["HUNYUAN_SECRET_KEY"] = keys.tx_key
os.environ["DASHSCOPE_API_KEY"] = keys.qianwen_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = keys.huggingface_key


def loadLLM(vendor):
    if vendor=="Baidu":
        model = QianfanChatEndpoint(
            endpoint="ernie-4.0-turbo-128k",
            temperature=0.1,
            timeout=120,
        )
    elif vendor=="Xunfei":
        #  科大讯飞星火默认的超时是30秒，不够，有些块的处理超过30秒，设置为120秒。
        model = ChatSparkLLM(
            spark_app_id = keys.appid,
            spark_api_key = keys.api_key,
            spark_api_secret = keys.api_secret,
            model='Spark4.0 Ultra',
            timeout=120
        )
    elif vendor=="Tengxun":
        model = ChatHunyuan(
            hunyuan_app_id=1303211952,
            hunyuan_secret_id = keys.tx_id,
            hunyuan_secret_key = keys.tx_key,
            model = "hunyuan-pro",
            # streaming=True
        )
    elif vendor=="Ali":
        model = ChatTongyi(model="qwen-max")   # qwen-plus, qwen-turbo, qwen-max
    elif vendor=="Ollama":
        model = ChatOllama(
            # qwen2.5:14b, MiniCPM3-4B-FP16, MFDoom/deepseek-r1-tool-calling:14b,
            # thirdeyeai/DeepSeek-R1-Distill-Qwen-7B-uncensored, deepseek-r1:14b
            model="qwen2.5:14b",  
            temperature=0,
            base_url="http://localhost:11434/"
            # other params...
        )
    elif vendor=="vLLM":
        model = ChatOpenAI(
            model="openbmb/MiniCPM3-4B",
            openai_api_key="token-abc123",
            openai_api_base="http://localhost:6000/v1",)
    elif vendor=="Siliconflow":
        # Pro/deepseek-ai/DeepSeek-V3, Pro/deepseek-ai/DeepSeek-R1
        model = ChatOpenAI(model="Pro/deepseek-ai/DeepSeek-R1", 
            api_key=keys.siliconflow_key,
            base_url = "https://api.siliconflow.cn/v1")     
    else:
        model = ChatOpenAI(model="gpt-4o")   # gpt-4o-mini, gpt-4o
    return model


def loadEmbedding(vendor):
    if vendor=="Baidu":
        embeddings = QianfanEmbeddingsEndpoint()
    elif vendor=="Xunfei":
        embeddings = SparkLLMTextEmbeddings(
            spark_app_id = keys.appid,
            spark_api_key = keys.api_key,
            spark_api_secret = keys.api_secret,
        )
    elif vendor=="Tengxun":
        embeddings = HunyuanEmbeddings()
    elif vendor=="Ali":
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v1", dashscope_api_key = keys.qianwen_key
        )
    elif vendor=="BAAI":
        # https://huggingface.co/docs/transformers/installation#offline-mode
        # 本地运行Embedding模型，设置为OFFLINE模式，不需要连接网络。
        # os.environ["HF_HUB_OFFLINE"] = "1"
        # os.environ["HF_DATASETS_OFFLINE"] = "1"
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", cache_folder="/home/demo/.cache/huggingface/hub/")
    elif vendor=="Ollama":
        embeddings = OllamaEmbeddings(model="bge-m3",base_url="http://localhost:11434")
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", )
    return embeddings


# if __name__ == "__main__":
#     query = "你好，请介绍一下你自己。"
#     chat_vendors = ["OpenAI","Baidu","Xunfei","Tengxun","Ali","Ollama","vLLM","Siliconflow"]
#     embedding_vendors = ["OpenAI","Baidu","Xunfei","Ali","BAAI","Ollama"]
# 
#     for vendor in chat_vendors:
#         try:
#             print("\nChat vendor: "+vendor+"\n")
#             chat = loadLLM(vendor)
#             res = chat.invoke(query)
#             print(res)
#         except Exception as e:
#             print(e)
# 
#     for vendor in embedding_vendors:
#         try:
#             print("\nEmbedding vendor: "+vendor+"\n")
#             embeddings = loadEmbedding(vendor)
#             single_vector = embeddings.embed_query(query)
                                                       
# query = "你好，请介绍一下你自己。"
# vendor = "vLLM"
# chat = loadLLM(vendor)
# res = chat.invoke(query)
# print(res)

# query = "你好，请介绍一下你自己。"
# vendor = "Ollama"
# embeddings = loadEmbedding(vendor)
# single_vector = embeddings.embed_query(query)
# print(len(single_vector))
# print(single_vector[:8])

