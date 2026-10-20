import os
from datetime import datetime
import json
from typing import Any
from urllib.request import Request
from fastapi import FastAPI, UploadFile
from ollama._types import ChatRequest
from pydantic import BaseModel
from starlette.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import logging
from Paper_Upload import add_pdf_to_database
from Vector_Database import make_chain
#连接static-新建会话写入文件-与AI交互-
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')

app = FastAPI(title="Thesis Assistant")
#挂载静态文件的存放目录 路径 存放目录 内部名字（自由定义）
app.mount("/static", StaticFiles(directory="static"), name="static")

#用chain与模型交互
chain = make_chain()

#创建会话存放目录sessions
if not os.path.exists("sessions"):
    os.makedirs("sessions")

#生成会话标识
def generate_session_id():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

#定义数据模型
class ApiResponse(BaseModel):
    code: int
    message:str
    data:Any

#定义数据模型
class ChatRequest(BaseModel):
    conversation_id:str
    question:str

#根据session_id获取文件名
def get_session_file_name(session_id):
    return f"sessions/{session_id}.json"


#定义路径操作函数
@app.get("/")
def root():
    return FileResponse("static/index.html")

#新建对话
@app.post("/api/sessions")
def create_session()->ApiResponse:
    print("session created")
    #生成会话标识-名字
    session_id=generate_session_id()
    #组装会话信息存入文件
    session_data={
        "current_session_id": session_id,
        "messages":[]
    }
    #写入文件
    with open("sessions/"+session_id+".json",'w',encoding="utf-8") as f:
        json.dump(session_data,f,ensure_ascii=False,indent=2)
    #返回数据
    return ApiResponse(code=200,message="会话创建成功",data=session_id)

#与大模型进行交互
@app.post("/api/chat")
def chat(request:ChatRequest)->ApiResponse:
    #先加载文件中的会话数据-与模型交互获得回答-更新消息列表-保存到文件
    session_path=get_session_file_name(request.conversation_id)
    with open(session_path,"r",encoding="utf-8") as f:
        session_data=json.load(f)
    history_messages=[]#历史会话 不包括当前
    for message in session_data["messages"]:
        history_messages.append(message)
    # 把历史消息转成文本
    history_text = "\n".join(
        f"{'用户' if item['role'] == 'user' else '助手'}：{item['content']}"
        for item in history_messages
    )
    result = chain.invoke({
        "question": request.question,
        "history": history_text,
    })
    # ChatDeepSeek 返回的 AIMessage 内容在 .content 里
    answer_text = result.content
    history_messages.append({"role": "user","content": request.question})
    history_messages.append({"role":"assistant","content":answer_text})
    session_data["messages"]=history_messages
    with open(session_path,"w",encoding="utf-8") as f:
        json.dump(session_data,f,ensure_ascii=False,indent=2)

    return ApiResponse(code=200,message="请求成功",data=answer_text)

#获取历史会话列表
@app.get("/api/sessions")
def get_sessions()->ApiResponse:
    #获取目录下文件名
    session_files=os.listdir("sessions")
    #获取会话id
    session_ids= [file.split(".")[0] for file in session_files]
    #倒序排序
    session_ids.sort(reverse=True)
    return ApiResponse(code=200,message="会话列表获取成功",data=session_ids)

#加载指定的历史会话
@app.get("/api/sessions/{session_id}")
def get_session(session_id:str)->ApiResponse:
    print("session found")
    #获取会话文件名-读取会话文件-返回数据
    session_file=get_session_file_name(session_id)
    with open(session_file,"r",encoding="utf-8") as f:
        session_data = json.load(f)
    return ApiResponse(code=200,message="加载会话成功",data=session_data)

#删除会话
@app.delete("/api/sessions/{session_id}")
def get_session(session_id:str)->ApiResponse:
    file_name=get_session_file_name(session_id)
    if os.path.exists(file_name):
       os.remove(file_name)
    return ApiResponse(code=200,message="删除成功",data=None)

#pdf上传处理
@app.post("/api/upload")
async def upload_file(file:UploadFile)->ApiResponse:
    file_bytes = await file.read()
    result = add_pdf_to_database(file.filename or "uploaded.pdf", file_bytes)

    return ApiResponse(code=200,message="上传成功",data=result)


#异常处理器
@app.exception_handler(Exception)
def handle_exception(request: Request, exc: Exception):
    logging.error(f"处理异常，请求路径：{request.url},异常信息：{exc}")
    return JSONResponse(content={"code":500,"message":"服务器内部错误","data":None})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)