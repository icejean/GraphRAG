import requests
# 这是Shiny for Python要用到的函数
from shiny import App, reactive, render, ui

# 定义UI函数，APP有两个参数，UI函数和server函数
# UI函数定义了Shiny APP在浏览器端的UI与行为。
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.script(ui.HTML("""
    //更新对话上下文并自动滚动到底部的JavaScript函数
    function scrollToBottom(message) {
      var textarea = document.getElementById('context');
      textarea.innerHTML = message;
      setTimeout(function() {
        window.scroll(0, document.body.scrollHeight);
      }, 200); // 可以根据需要调整延迟时间
    }
    //监听服务器发回的对话信息
    Shiny.addCustomMessageHandler('scrollToBottom', scrollToBottom);
    
    // 显示溯源信息，点击溯源信息上的链接后，通知服务器查询知识图谱中的信息。
    function showInfo(info) {
      //更新查看的知识图谱语料来源
      var sourceId = document.getElementById('sourceId');
      sourceId.value = info
      //发送文本块或社区的ID到服务器端查询。
      Shiny.setInputValue('sourceId', info, {priority: 'event'});      
    }

    // 演示性质，先用最简单的方式显示溯源的知识图谱语料。
    function showSource(message) {
      var sourcetext = document.getElementById('sourcetext');
      sourcetext.innerHTML = message;
    }
    //监听服务器发回的溯源信息
    Shiny.addCustomMessageHandler('showSource', showSource);
      """))
    ),
    
    ui.HTML("<h3><b>知识图谱GraphRAG聊天Agent</b></h3>"),
    
    ui.navset_tab(
        ui.nav_panel("知识图谱对话", 
           ui.HTML("""
说明一：这个GraphRAG示例Shiny APP选取了网络小说《悟空传》的前7章建立知识图谱来回答问题。
对话中Agent会根据所提的问题选择是否查询知识图谱，如果是全局性的问题，会使用知识图谱中预先生成的社区摘
要来回答问题；如果是局部性的问题，会在知识图谱中查找最相似的问题，并找到最相关的其它材料来回答问题。
因为是要求准确的回答，LLM受到指示，如果不知道答案就回答不知道。所以RAG实际上是Prompt Engineering
的一种形式，它有效克服了LLM的幻觉、知识缺失和知识截止的问题，GraphRAG则通过使用知识图谱检索逻辑上深度相关的
附加材料，让LLM的回答更准确更有深度，更符合现实世界的客观事实与逻辑，更适用于法律、金融、医疗、教育等
准确率要求很高的领域。<br>
例如：<br>
1、不需要检索知识图谱的普通聊天：<br>
你好吗？<br>
2、要检索知识图谱的全局查询问题：<br>
《悟空传》书中的主要人物有哪些？<br>
唐僧的取经团队对路上遇到的妖怪采取什么样的态度？<br>
3、要检索知识图谱的局部查询问题：<br>
孙悟空跟女妖之间有什么故事？<br>
唐僧和会说话的树讨论了什么？<br>
说明二：对话输出中在知识图谱的引用上有超链接，点击可以溯源查验。如果是引用了社区，就在“溯源查验”页中显示社区的
摘要与全文，如果是引用了文本块，就显示文本块的内容。有可能出现的情况是，只有1个实体结点的社区没有
做摘要，会显示“在知识图谱中没有检索到该语料”。<br>
说明三：LangGraph的删除消息功能目前还在Beta阶段，暂不支持清空整个对话，至少要保留第1轮对话。<br>
说明四：在对话开始之前，可以从页面底部的下拉列表中选择使用的LLM厂商。目前测试过的LLM中，
国产的LLM只有阿里与百度在LangChain的集成中实现了工具绑定与调用的功能。<br>
说明五：后端通过FastAPI提供RESTful API Agent调用，需要先登录FastAPI服务器，点击<登录>按钮用默认的
用户名与口令登录后才能进行对话等操作。点击<登出>退出登录并清空当前session的对话。<br>
           """),
           ui.HTML("<h4><b>对话记录</b></h4>"),
           ui.HTML(' <div id="context" style="width:100%;resize:vertical;"> </div>'),
           ui.tags.h6(" "),
           ui.input_text_area("prompt","输入：",width="100%", rows =4, resize="vertical", value =""), 
           ui.HTML("<table><tr><td>"),
           ui.input_action_button("sendout", "提交", class_ = "btn-success"),
           ui.HTML("</td><td>&nbsp</td><td>"),
           ui.input_action_button("clearContext", "清空对话", class_ = "btn-clear"),
           ui.HTML("</td><td>&nbsp</td><td>"),
           ui.input_select(
               'vendor',
               '',
               {"OpenAI": "OpenAI",
                 "Ali": "阿里",
                 "Baidu": "百度",
                 "Ollama": "Ollama",
                 "vLLM": "vLLM",
                 "Siliconflow" : "硅基流动",
               },
               selected = 'Ollama'
           ),
           ui.HTML("</td><td>&nbsp</td><td>"),
           ui.input_action_button("login", "登录", class_ = "btn-success"),
           ui.HTML("</td><td>&nbsp</td><td>"),
           ui.input_action_button("logout", "登出", class_ = "btn-clear"),
           ui.HTML("</td></tr></table>"),
           ui.tags.h6(" "),
             
        ),
        ui.nav_panel("溯源查验", 
             ui.HTML("<table><tr><td>"),
             ui.HTML("查看的知识图谱来源：</td><td>"),
             ui.input_text("sourceId","", value="", width = "50ch"),
             ui.HTML("</td></tr></table>"),
             # 插入javascript，禁止自己修改 sourceId
             ui.tags.script(ui.HTML("""
              var context = document.getElementById('sourceId');
              context.readOnly = true;
             """)),
             ui.HTML(' <div id="sourcetext" style="width:100%;resize:vertical;"> </div>'),
        ),
        id="functionTabs",
    ),
    
)

# FastAPI提供各种功能的URL
login_url = "http://117.50.174.65/wukong/login"
logout_url = "http://117.50.174.65/wukong/logout"
setllm_url = "http://117.50.174.65/wukong/setllm"
chat_url = "http://117.50.174.65/wukong/chat"
history_url = "http://117.50.174.65/wukong/history"
reset_url = "http://117.50.174.65/wukong/reset"
source_url = "http://117.50.174.65/wukong/getsource"
# 默认的用户名、口令，演示性质。
user = "jean"
password = "demo"

# 该函数调用FastAPI登录FastAPI服务器。
def login(url, username, password):
    # 封装要发送的数据
    data = {
        'username': username,
        'password': password
    }
    # 发送POST请求
    response = requests.post(url, json=data)
    # 检查请求是否成功
    if response.status_code == 200:
        print('登录成功')
        # 提取session_token并返回
        response_data = response.json()  # 获取JSON解析后的数据
        # 输出相关返回信息
        print('返回的消息:', response_data.get('message', '无返回消息'))
        session_token = response_data.get('session_token')
        if session_token:
            print('获取到的session_token:', session_token)
            return session_token
        else:
            print('未获取到session_token')
            return None
    else:
        print('登录失败，状态码:', response.status_code)
        return None

# 该函数调用FastAPI登出FastAPI服务器。
def logout(url, username, session_token):
    # 设置请求头
    headers = {
        'UserName': username,
        'Authorization': session_token
    }
    # 发送GET请求
    response = requests.get(url, headers=headers)
    # 检查响应状态
    if response.status_code == 200:
        print('成功登出')
        response_data = response.json()  # 获取JSON解析后的数据
        # 输出相关返回信息
        print('返回的消息:', response_data.get('message', '无返回消息'))
        return True
    else:
        print('登出失败，状态码:', response.status_code)
        return False

# 该函数调用FastAPI进行连续的对话。
def chat(url, username, session_token, query):
    # 配置HTTP头部信息
    headers = {
        'UserName': username,
        'Authorization': session_token
    }
    # 封装发送的JSON数据
    data = {'query': query}
    # 发送POST请求
    response = requests.post(url, headers=headers, json=data)
    # 检查响应状态并处理
    if response.status_code == 200:
        print('请求成功')
        # FastAPI返回的是纯文本
        messages = response.text
        # print('返回的消息:', messages)
        return messages
    else:
        print('请求失败，状态码:', response.status_code)
        return None
    
# 该函数调用FastAPI返回sesion的对话记录。
def history(url, username, session_token):
    # 配置HTTP头部信息
    headers = {
        'UserName': username,
        'Authorization': session_token
    }
    # 发送POST请求
    response = requests.get(url, headers=headers)
    # 检查响应状态并处理
    if response.status_code == 200:
        print('请求成功')
        # FastAPI返回的是纯文本
        messages = response.text
        # print('返回的消息:', messages)
        return messages
    else:
        print('请求失败，状态码:', response.status_code)
        return None

# 该函数调用FastAPI清空session对话，只留下第一轮对话。
def reset(url, username, session_token):
    # 设置请求头
    headers = {
        'UserName': username,
        'Authorization': session_token
    }
    # 发送GET请求
    response = requests.get(url, headers=headers)
    # 检查响应状态
    if response.status_code == 200:
        print('请求成功')
        response_data = response.json()  # 获取JSON解析后的数据
        # 输出相关返回信息
        print('返回的消息:', response_data.get('message', '无返回消息'))
        return True
    else:
        print('请求失败，状态码:', response.status_code)
        return False

# 该函数调用FastAPI，设置要使用的LLM厂商，切换后端使用的LLM。   
def setllm(url, username, session_token, vendor):
    # 配置HTTP头部信息
    headers = {
        'UserName': username,
        'Authorization': session_token
    }
    # 封装发送的JSON数据
    data = {"vendor": vendor}
    # 发送POST请求
    response = requests.post(url, headers=headers, json=data)
    # 检查响应状态并处理
    if response.status_code == 200:
        print('请求成功')
        response_data = response.json()  # 获取JSON解析后的数据
        # 输出相关返回信息
        message = response_data.get('message', '无返回消息')
        print('返回的消息:', message)
        return True
    else:
        print('请求失败，状态码:', response.status_code)
        return False

# 该函数调用FastAPI，返回溯源查验用的原始材料。
def getsource(url, username, session_token, sourceId):
    # 配置HTTP头部信息
    headers = {
        'UserName': username,
        'Authorization': session_token
    }
    # 封装发送的JSON数据
    data = {'sourceId': sourceId}
    # 发送POST请求
    response = requests.post(url, headers=headers, json=data)
    # 检查响应状态并处理
    if response.status_code == 200:
        print('请求成功')
        # 返回的是纯文本。
        messages = response.text
        # print('返回的消息:', messages)
        return messages
    else:
        print('请求失败，状态码:', response.status_code)
        return None

# 定义服务器函数
# 服务器函数定义了Shiny APP在服务器端的业务逻辑与行为。
def server(input, output, session):
    # 登录FastAPI服务器后得到的session_token存放在Shiny APP的session变量中
    # Shiny server与FastAPI server是松耦合的分布式服务器，各自维持自己的session。
    # Shiny server中通过session_token来标记session，
    # FastAPI服务器中Agent通过当前用户current_user来维持session。
    # session_token就是他们之间属于同一个session的共同联系，
    # 通过FastAPI服务器中的字典login_users链接，key是current_user，value是session_token。
    def get_session_token():
        try:
            session_token = session.session_token
        except Exception as e:
            session_token = None
        return  session_token
  
    # 处理登录操作
    @reactive.Effect
    @reactive.event(input.login)
    def _():
        session_token = get_session_token()
        if not session_token:
            session_token = login(login_url,user,password)
            session.session_token = session_token
            ui.update_text_area(id = "prompt", value = "用户 "+user+" 登录成功！")
        else:
            ui.update_text_area(id = "prompt", value = "用户已经登录！")

    # 处理登出操作
    @reactive.Effect
    @reactive.event(input.logout)
    def _():
        session_token = get_session_token()
        if session_token:
            logout(logout_url, user, session.session_token)
            del session.session_token
            ui.update_text_area(id = "prompt", value = "用户 "+user+" 登出成功！")
        else:
            ui.update_text_area(id = "prompt", value = "用户未登录！")

    # 处理清空对话按钮
    # 注意，调用session.send_custom_message()要用异步的方式
    # 参阅：https://shiny.posit.co/py/api/core/Session.html#shiny.session.Session.send_custom_message
    @reactive.Effect
    @reactive.event(input.clearContext)
    async def _():
        # 更新到浏览器端的输入中显示
        session_token = get_session_token()
        if not session_token:
          return
        reset(reset_url, user, session_token)
        ui.update_text_area(id = "prompt", value = "")
        ui.update_text(id = "sourceId", value = "")
        await session.send_custom_message(type = 'scrollToBottom', message = "")

    # 处理提交按钮
    @reactive.Effect
    @reactive.event(input.sendout)
    async def _():
        print(input.prompt())
        # 需要较长时间，在浏览器端显示正在执行的信息。
        id = ui.notification_show("正在询问Agent...", duration=None)
        session_token = get_session_token()
        if not session_token:
          ui.update_text_area(id = "prompt", value = "请先登录后再提问。")
          return
        messages = chat(chat_url, user, session_token, input.prompt())
        # print(messages)
        if messages:
            # 更新到浏览器端的输入中显示
            ui.update_text_area(id = "prompt", value = "")
            ui.update_text(id = "sourceId", value = "")
            await session.send_custom_message(type = 'scrollToBottom', message = messages)
        
        ui.notification_remove(id)

    # 处理选择使用的LLM厂商
    @reactive.Effect
    @reactive.event(input.vendor)
    def _():
        print(input.vendor())
        if len(input.vendor())>0:
            session_token = get_session_token()
            if not session_token:
               ui.update_text_area(id = "prompt", value = "请先登录后再选择LLM厂商。")
               return
            setllm(setllm_url, user, session_token, input.vendor())
            ui.update_text_area(id = "prompt", value = "LLM 已切换至："+input.vendor())
        else:
            print("空的厂商名！")

    # 处理查询溯源信息
    @reactive.Effect
    @reactive.event(input.sourceId)
    async def _():
        print(input.sourceId())
        if len(input.sourceId())>0:
            session_token = get_session_token()
            if not session_token:
               ui.update_text_area(id = "prompt", value = "请先登录后再操作。")
               return
            res = getsource(source_url, user, session_token, input.sourceId())
            await session.send_custom_message(type = 'showSource', message = res)
            ui.update_navs("functionTabs", selected="溯源查验")
        else:
            print("空的信息源！")

# UI函数与服务器函数组合成Shiny APP
app = App(app_ui, server, debug=False)
