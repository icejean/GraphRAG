import os
import time
from keys import keys

# 用LangSmith记录LLM调用的日志，可在在https://smith.langchain.com上查看详情
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = keys.langchain_api_key

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_ollama import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA


def loadLLM(vendor):
    if vendor=="DeepSeek":  # DeepSeek官方
        model = ChatOpenAI(
            # deepseek-reasoner, deepseek-chat
            model="deepseek-reasoner",   
            api_key=keys.deepseek_key,
            base_url = "https://api.deepseek.com/v1"
        )
    elif vendor=="Siliconflow":  # 硅基流动
        model = ChatOpenAI(
            # deepseek-ai/DeepSeek-V3, deepseek-ai/DeepSeek-R1
            # Pro/deepseek-ai/DeepSeek-V3, Pro/deepseek-ai/DeepSeek-R1
            model="Pro/deepseek-ai/DeepSeek-R1",
            api_key=keys.siliconflow_key,
            base_url = "https://api.siliconflow.cn/v1"
        )
    elif vendor=="Tengxun":  # 腾讯云
        model = ChatOpenAI(
            # deepseek-r1, deepseek-v3
            model="deepseek-r1",
            api_key=keys.tengxun_key,
            base_url = "https://api.lkeap.cloud.tencent.com/v1"
        )
    elif vendor=="Telecom":  # 天翼云
        model = ChatOpenAI(
            model="4bd107bff85941239e27b1509eccfe98",  # deepseek-r1
            # model="9dc913a037774fc0b248376905c85da5",  # deepseek-v3
            api_key=keys.telecom_key,
            base_url = "https://wishub-x1.ctyun.cn/v1/"
        )
    elif vendor=="Huawei":  # 华为云 注意DeepSeek-R1与DeepSeek-V3的base_url不同
        model = ChatOpenAI(
            model="DeepSeek-R1",  # DeepSeek-R1 32K
            # model="DeepSeek-V3",  # DeepSeek-V3 32K
            # max_tokens=8192,
            api_key=keys.huawei_key,
            base_url = "https://infer-modelarts-cn-southwest-2.modelarts-infer.com/v1/infers/952e4f88-ef93-4398-ae8d-af37f63f0d8e/v1" #DeepSeek-R1 32K
            # base_url = "https://infer-modelarts-cn-southwest-2.modelarts-infer.com/v1/infers/fd53915b-8935-48fe-be70-449d76c0fc87/v1" #DeepSeek-V3 32K
            # base_url = "https://infer-modelarts-cn-southwest-2.modelarts-infer.com/v1/infers/861b6827-e5ef-4fa6-90d2-5fd1b2975882/v1"  #DeepSeek-R1 8K
            # base_url = "https://infer-modelarts-cn-southwest-2.modelarts-infer.com/v1/infers/707c01c8-517c-46ca-827a-d0b21c71b074/v1" #DeepSeek-V3 8K
        )
    elif vendor=="Baidu": # 百度云
        model = ChatOpenAI(
            # deepseek-r1, deepseek-v3
            model="deepseek-r1",
            api_key=keys.baidu_key,
            base_url = "https://qianfan.baidubce.com/v2"
        )
    elif vendor=="Ali": # 阿里云
        model = ChatOpenAI(
            # deepseek-r1, deepseek-v3
            model="deepseek-r1",
            api_key=keys.ali_key,
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    elif vendor=="Bytedance": # 字节跳动
        model = ChatOpenAI(
            model="ep-20250210064226-vk6vk",  # deepseek-r1
            # model="ep-20250210065815-wzw6j",  # deepseek-v3
            api_key=keys.volcengin_key,
            base_url = "https://ark.cn-beijing.volces.com/api/v3"
        )
    elif vendor=="Nvidia": # 英伟达
        model = ChatNVIDIA(
            model="deepseek-ai/deepseek-r1",
            api_key=keys.nvidia_key,
            temperature=0.1,
            max_tokens=8192       
        ) 
    elif vendor=="Xunfei": # 科大讯飞
        model = ChatOpenAI(
            model="xdeepseekr1",   
            # model="xdeepseekv3",   
            api_key=keys.xunfei_key,
            base_url = "https://maas-api.cn-huabei-1.xf-yun.com/v1",          
            temperature=0.1,
            max_tokens=8192,
            streaming=True,
            timeout=1200
        )
    elif vendor=="Sensecore": # 商汤万象
        model = ChatOpenAI(
            model="DeepSeek-R1",   # DeepSeek-R1,  DeepSeek-V3
            api_key=keys.sencecore_key,
            base_url = "https://api.sensenova.cn/compatible-mode/v1/", 
            max_tokens=8192
        )
    else:   # Ollama本地运行
        model = ChatOllama(
            # deepseek-r1-7b-fp16,  deepseek-r1, deepseek-r1:14b, deepseek-r1:32b
            model="thirdeyeai/DeepSeek-R1-Distill-Qwen-7B-uncensored",
            temperature=0.1,
            base_url="http://localhost:11434/"
        )
    return model


# llm = loadLLM("DeepSeek")
# llm = loadLLM("Siliconflow")
# llm = loadLLM("Tengxun")
# llm = loadLLM("Telecom")
llm = loadLLM("Huawei")
# llm = loadLLM("Baidu")
# llm = loadLLM("Ali")
# llm = loadLLM("Bytedance")
# llm = loadLLM("Nvidia")
# llm = loadLLM("Xunfei")
# llm = loadLLM("Sensecore")


t1=time.time()
answer = llm.invoke("你好，请介绍一下你自己。")
t2=time.time()
print(answer)
print(t2-t1)

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

# 这一段天翼云内容审查不通过： 'code': '600003','type': 'TEXT_AUDIT_QUESTION_NOT_PASS'
input_text="""
《悟空传》今何在2017-07-11一个声音狂笑着，他大笑着殴打神仙，大笑着毁灭一切，他知道神永远杀不完，他知道天宫无边无际。这战斗将无法终止，直到他倒下，他仍然狂笑，笑出了眼泪。这个天地，我来过，我奋战过，我深爱过，我不在乎结局。[01.]四个人走到这里，前边一片密林，又没有路了。“悟空，我饿了，给我找些吃的来。”唐僧往石头上大模大样一坐，命令道。“我正忙着，你不会自己去找？又不是没有腿。”孙悟空拄着棒子说。“你忙？忙什么？”“你不觉得这晚霞很美吗？”孙悟空说，眼睛还望着天边，“我只有看看这个，才能每天坚持向西走下去啊。”“你可以一边看一边找啊，只要不撞到大树上就行。”“我看晚霞的时候不做任何事！”“孙悟空你不能这样，不能这样欺负秃头，你把他饿死了，我们就找不到西天，找不到西天，我们身上的诅咒永远也解除不了。”猪八戒说。“呸！什么时候轮到你这个猪头说话了？”“你说什么？你说谁是猪？！”“不是，是猪头！啊哈哈哈……”“你敢再说一遍！”猪八戒举着钉耙就要往上冲。“吵什么吵什么！老子要困觉了！要打滚远些打！”沙和尚大吼。三个恶棍怒目而视。“打吧打吧，打死一个少一个。”唐僧站起身来，“你们是大爷，我去给你们找吃的，还不行吗？最好让妖怪吃了我，那时你们就哭吧。”“快去吧，那儿有女妖精正等着你呢。”孙悟空叫道。“哼哼哼哼……”三个怪物都在冷笑。“别以为我离了你们就不行！”唐僧回头冲他们挥挥拳头，拍拍身上的尘土，又整整长袍，开始向林中走去。刚迈一步，“刺啦”——僧袍就被扯破了。“哈哈哈哈……”三个家伙笑成一团，也忘了打架。[02.]这是一片紫色的丛林，到处长着奇怪的植物，飘着终年不散的青色雾气，越往里走，脚下就越潮湿，头上就越昏暗，最后枝叶完全遮蔽了天空，唐僧也完全迷路了。
"""

# # 天翼云要用这一段
# input_text="""
# 她第二次扬起手，漫天的银尘开始旋转，绕着她和天蓬所在的地方，它们越转越快，越转越快，最后变成了一个无比巨大的银色光环。天蓬要被这奇景惊喜得晕倒了，他脚步踉跄，不由得微微靠在了她身上。她并没有推开他，而是用手轻轻挽住天蓬，“小心。”她仍然是那么轻声地说。这两个字是天蓬八十万年来听到的最美的音乐。她第三次扬起手，光环开始向中心汇聚，形成亿万条向核心流动的银线，光环中心，一个小银核正越来越清晰。“是什么在吸引它们？”天蓬问。“是我。”她说。……“是我们。”她笑了，用手指轻轻点了一下天蓬。天蓬觉得那银色河流也在这一触间随他的血脉流遍了全身，他无法自抑，将她揽入怀中。他们深吻着，几十万年等待的光阴将这一刻铸成永恒。当长吻终于结束的时候，她从他的怀里脱身而出，一看天际，忽然惊叫了起来：“糟了！”她被吻时法力消散，银核已经汇聚，却还有几亿颗散落在天河各处。她掩面哭泣了起来：“我做了那么久，花了那么长的时间，还是失败了。”天蓬轻轻揽住她的肩：“别哭了，世间没有一件造物会是完美的，但有时缺憾会更美。你抬头看看。”她抬起头，只见天河四野，俱是银星闪耀。“从前天河是一片黑暗的，现在你把它变成了银色的，那么，我们就改叫它‘银河’吧，那个银色的小球，我们就叫它……”“用我的名字吧，叫它——月。”“月……那我可以说……月光下，映着一对爱人吗？”……月光下，映着一对爱人，他们紧紧相拥。
# """

tuple_delimiter = " : "
record_delimiter = "\n"
completion_delimiter = "\n\n"

entity_types = ["人物","妖怪","位置"]
relationship_types=["师徒", "师兄弟", "对抗", "对话", "态度", "故事地点", "其它"]
chat_history = []

t1 = time.time()
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
print(input_text)
print("\n")
print(answer.content)
print("块耗时：",t2-t1,"秒")
print("\n")

