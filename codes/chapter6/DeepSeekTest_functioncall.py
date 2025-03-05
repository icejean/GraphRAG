import os
import time
from keys import keys

# 用LangSmith记录LLM调用的日志，可在在https://smith.langchain.com上查看详情
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = keys.langchain_api_key

os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)


def loadLLM(vendor):
    if vendor=="DeepSeek":  # DeepSeek官方
        model = ChatOpenAI(
            # deepseek-reasoner, deepseek-chat
            model="deepseek-chat",   
            api_key=keys.deepseek_key,
            base_url = "https://api.deepseek.com/v1"
        )
    elif vendor=="Siliconflow":  # 硅基流动
        model = ChatOpenAI(
            # deepseek-ai/DeepSeek-V3, deepseek-ai/DeepSeek-R1
            # Pro/deepseek-ai/DeepSeek-V3, Pro/deepseek-ai/DeepSeek-R1
            model="Pro/deepseek-ai/DeepSeek-V3",
            api_key=keys.siliconflow_key,
            base_url = "https://api.siliconflow.cn/v1"
        )
    elif vendor=="Tengxun":  # 腾讯云
        model = ChatOpenAI(
            # deepseek-r1, deepseek-v3
            model="deepseek-v3",
            api_key=keys.tengxun_key,
            base_url = "https://api.lkeap.cloud.tencent.com/v1"
        )
    elif vendor=="Telecom":  # 天翼云
        model = ChatOpenAI(
            # model="4bd107bff85941239e27b1509eccfe98",  # deepseek-r1
            model="9dc913a037774fc0b248376905c85da5",  # deepseek-v3
            api_key=keys.telecom_key,
            base_url = "https://wishub-x1.ctyun.cn/v1/"
        )
    elif vendor=="Huawei":  # 华为云 注意DeepSeek-R1与DeepSeek-V3的base_url不同
        model = ChatOpenAI(
            # model="DeepSeek-R1",  # DeepSeek-R1 32K
            model="DeepSeek-V3",  # DeepSeek-V3 32K
            # max_tokens=8192,
            api_key=keys.huawei_key,
            # base_url = "https://infer-modelarts-cn-southwest-2.modelarts-infer.com/v1/infers/952e4f88-ef93-4398-ae8d-af37f63f0d8e/v1" #DeepSeek-R1 32K
            base_url = "https://infer-modelarts-cn-southwest-2.modelarts-infer.com/v1/infers/fd53915b-8935-48fe-be70-449d76c0fc87/v1" #DeepSeek-V3 32K
        )
    elif vendor=="Baidu": # 百度云
        model = ChatOpenAI(
            # deepseek-r1, deepseek-v3
            model="deepseek-v3",
            api_key=keys.baidu_key,
            base_url = "https://qianfan.baidubce.com/v2"
        )
    elif vendor=="Ali": # 阿里云
        model = ChatOpenAI(
            # deepseek-r1, deepseek-v3
            model="deepseek-v3",
            api_key=keys.ali_key,
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    elif vendor=="Bytedance": # 字节跳动
        model = ChatOpenAI(
            # model="ep-20250210064226-vk6vk",  # deepseek-r1
            model="ep-20250210065815-wzw6j",  # deepseek-v3
            api_key=keys.volcengin_key,
            base_url = "https://ark.cn-beijing.volces.com/api/v3"
        )
    elif vendor=="Xunfei": # 科大讯飞
        model = ChatOpenAI(
            # model="xdeepseekr1",   
            model="xdeepseekv3",   
            api_key=keys.xunfei_key,
            base_url = "https://maas-api.cn-huabei-1.xf-yun.com/v1",          
            temperature=0.1,
            max_tokens=8192,
            streaming=True,
            timeout=1200
        )
    elif vendor=="Sensecore": # 商汤万象
        model = ChatOpenAI(
            model="DeepSeek-V3",   # DeepSeek-R1,  DeepSeek-V3
            api_key=keys.sencecore_key,
            base_url = "https://api.sensenova.cn/compatible-mode/v1/", 
            max_tokens=8192
        )
    return model



# llm = loadLLM("DeepSeek")
# llm = loadLLM("Siliconflow")
# llm = loadLLM("Tengxun")
# llm = loadLLM("Telecom")
# llm = loadLLM("Huawei")
# llm = loadLLM("Baidu")
# llm = loadLLM("Ali")
# llm = loadLLM("Bytedance")
# llm = loadLLM("Xunfei")
# llm = loadLLM("Sensecore")


from langchain_core.tools import tool

# 定义一个工具函数
@tool
def greeting(name: str) ->str : 
    '''向朋友致欢迎语'''
    return f"你好啊, {name}"
 
chat_vendors = ["DeepSeek","Siliconflow","Tengxun","Telecom","Huawei","Baidu","Ali","Bytedance","Xunfei","Sensecore"]

for vendor in chat_vendors:
    try:
        t1 = time.time()
        print("\nVendor: "+vendor+"\n")
        llm = loadLLM(vendor)
        llm_with_tool = llm.bind_tools([greeting])
        resp = llm_with_tool.invoke("你好，我是Jean，问候一下。")
        t2 = time.time()
        print(t2-t1,"\n")
        print(resp)
    except Exception as e:
        print(e)

