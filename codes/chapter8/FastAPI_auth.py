# python FastAPI_auth.py >> /var/log/pywukong.log 2>&1 &

# https://github.com/fastapi/fastapi

# 三分钟快速搭建基于FastAPI的AI Agent应用
# https://www.163.com/dy/article/J5NBDLCV0552NMWZ.html

# 这是一个使用 FastAPI 框架编写的简单应用程序的示例。 
# 加载《悟空传》查询Agent
import sys
import json
import logging
# KGGraphRAGAgent.py从这个目录加载。
sys.path.append("/home/jean/scripts")
from KGGraphRAGAgent import ask_agent, clear_session, get_messages, select_vendor
from KGGraphRAGAgent import format, add_links_to_text, get_source
# 导入FastAPI模块 
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware.sessions import SessionMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import secrets

# 设置日志记录运行时的异常
logging.basicConfig(level=logging.INFO, filename='/var/log/pywukong_runtime.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 创建一个FastAPI应用实例
# How to Access FastAPI SwaggerUI Docs Behind an NGINX proxy?
# https://stackoverflow.com/questions/67435296/how-to-access-fastapi-swaggerui-docs-behind-an-nginx-proxy
# 在nginx中配置反向代理访问FastAPI的URI是"wukong"
# 要指定该root_path，才能正确访问API文档
# http://117.50.174.65/wukokng/redoc
# 或 http://117.50.174.65/wukokng/docs
app = FastAPI(root_path="/wukong")

# 添加CORS中间件，浏览器中允许Javascript跨域访问，从其它域的javascript中调用本域的URL
app.add_middleware(
    CORSMiddleware,
    # 根据需要修改允许的来源， "*"， "http://example.com", ["http://localhost"]
    # "*"就是允许所有来源，此时不能带cookie，带cookie的话不能允许 "*"
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 由于FastAPI是基于Starlette的，可以用Starlette的SessionMiddleware来管理简单的会话。
# 这不需要额外的库安装，因为Starlette已内置在FastAPI中。
# 设置一个密钥，用于加密会话cookie。在生产环境中，请使用复杂的随机密钥。
# app.add_middleware(SessionMiddleware, secret_key="!secret")
# Javascript跨域访问FastAPI API通过allow_origins=["*"]实现，
# 此时无法通过cookie来维持session，需要自己管理session。
# 记录已登录用户的全局字典，内容是用户名与临时生成的access token，用于管理session。
login_users = {}


# /login路由的输入
class LoginRequest(BaseModel):
    username: str
    password: str

# /chat路由的输入
class ChatRequest(BaseModel):
    query: str

# /setllm路由的输入
class LlmRequest(BaseModel):
    vendor: str

# /getsource路由的输入
class SourceRequest(BaseModel):
    sourceId: str

# 临时的用户数据来验证登录，生产环境要连接用户的验证系统。
users = {
    "jean": "demo",
}

# 验证用户名密码
def verify_username_password(username: str, password: str):
    if username in users and users[username] == password:
        return True
    return False

# 返回当前登录的用户名
# 登录后用户的 http request header中包含UserName与Authorization，生产环境要用https加密通讯。
def get_current_user(request: Request):
    username : str = request.headers.get('UserName')
    authorization: str = request.headers.get('Authorization')
    if not login_users[username]:
        raise HTTPException(status_code=401, detail="You are not logged in.")
    if authorization !=  login_users[username] :
        raise HTTPException(status_code=400, detail="Invalid username or password.")
    return username


# 定义全局异常处理器，捕获并记录任何未处理的异常。
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # 记录异常到日志文件
    logging.error(f"Unexpected error occur: {exc}, from: {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. Our team is working on it."},
    )


# 定义一个路由，当访问'/'时会被触发 
# http://117.50.174.65/wukokng
@app.get("/")
# 定义一个函数，返回一个字典，key为"Hello"，value为"Banks!" 
def read_root():
    try:
        return {"Hello": "World!"}
    except Exception as e:
        logging.error(f"Error in /: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error in root API")
      

# 登录 路由： http://117.50.174.65/wukokng/login
@app.post("/login")
def login(request: Request, login_request: LoginRequest):
    try:
        if verify_username_password(login_request.username, login_request.password):
            # 生成一个安全的会话令牌，后续调用中用户要提供。
            session_token = secrets.token_urlsafe()
            # 记录已登录的用户
            login_users[login_request.username] = session_token
            # 没有登录过的用户插入第1轮对话的消息，因为LangGraph Agent框架目前无法完全清空对话
            # 这样同一个用户下一个session登录时，遗留的就是这一条打招呼的信息。
            # 第一次登录时插入一条这样的对话，就不会影响后续的各次登录与对话。
            # try:
            #     messages = get_messages(login_request.username)
            # except Exception as e:
            #     messages = None
            # if not messages or len(messages)==0:
            #     messages = ask_agent("你好，请严格根据要求回答问题。", login_request.username)
            # 创建响应并设置session令牌到cookie
            response = JSONResponse(content={"message": f"Welcome {login_request.username}!",\
                "session_token": session_token, "username":login_request.username})
            return response
        else:
            raise HTTPException(status_code=400, detail="Invalid username or password.")
    except Exception as e:
        logging.error(f"Error in /login: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error in /login API")
          

# 登出 路由： http://117.50.174.65/wukokng/logout
# 需要登录后才能执行： current_user: str = Depends(get_current_user)
@app.get("/logout")
def logout(request: Request, current_user: str = Depends(get_current_user)):
    try:
        # 登出时清空当前的对话
        clear_session(current_user)
        # 从已登录用户字典中删除
        del login_users[current_user]
        return {"message": f"User {current_user} have been logged out."}
    except Exception as e:
        logging.error(f"Error in /logout: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error in /logout API")
      

# 调用Agent对话 路由：http://117.50.174.65/wukokng/chat
# 需要登录后才能执行： current_user: str = Depends(get_current_user)
@app.post("/chat")
def chat(chat_request: ChatRequest, current_user: str = Depends(get_current_user)):
    try:
        # 此处返回的是整个session的对话记录，已经拼接成字符串并作了HTML格式处理。
        messages = ask_agent(chat_request.query, current_user)
        # 返回纯文本，因为加了朔源链接等HTML标记。
        return Response(content=messages, media_type="text/plain")
    except Exception as e:
        logging.error(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error in /chat API")
      
      
# 设置使用的LLM厂商 路由：http://117.50.174.65/wukokng/setllm
# 需要登录后才能执行： current_user: str = Depends(get_current_user)
@app.post("/setllm")
def setllm(llm_request: LlmRequest, current_user: str = Depends(get_current_user)):
    try:
        select_vendor(llm_request.vendor)
        return {"message": f"LLM has been changed to  {llm_request.vendor}."}
    except Exception as e:
        logging.error(f"Error in /setllm: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error in /setllm API")
      

# 清空session对话 路由：http://117.50.174.65/wukokng/reset
# 需要登录后才能执行： current_user: str = Depends(get_current_user)
# LanagGraph目前的实现无法全部清空session，会留下第一轮对话。
@app.get("/reset")
def reset(current_user: str = Depends(get_current_user)):
    try:
        clear_session(current_user)
        return {"message": f"User {current_user}'s session is cleared and reset."}
    except Exception as e:
        logging.error(f"Error in /reset: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error in /reset API")
      

# 获取对话历史 路由：http://117.50.174.65/wukokng/history
# 需要登录后才能执行： current_user: str = Depends(get_current_user)
@app.get("/history")
def history(current_user: str = Depends(get_current_user)):
    try:
        messages = get_messages(current_user)
        # 拼接成字符串，进行HTML格式化。
        response = format(messages)
        # 为查询返回的数据来源引用加上超链接。
        response = add_links_to_text(response)
        # 返回纯文本，因为加了溯源链接等。
        return Response(content=response, media_type="text/plain")
    except Exception as e:
        logging.error(f"Error in /history: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error in /history API")
      

# 查看溯源材料 路由：http://117.50.174.65/wukokng/getsource
# 需要登录后才能执行： current_user: str = Depends(get_current_user)
@app.post("/getsource")
def getSource(source_request:SourceRequest, current_user: str = Depends(get_current_user)):
    try:
        response = get_source(source_request.sourceId)
        # 返回纯文本，因为原始材料是HTML格式。
        return Response(content=response, media_type="text/plain")
    except Exception as e:
        logging.error(f"Error in /getsource: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error in /getsource API")
      

# 如果主程序为 __main__，则启动服务器 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8010)
  
