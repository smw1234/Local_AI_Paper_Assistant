"""
将原先chain调用改为agent
"""
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from Vector_Database import qa_model, retrieve_context_with_history

# 获取当前 Python 文件所在目录。
# MCP Server 和 Agent 文件放在项目根目录时，这样就能定位 MCP_Server.py。
PROJECT_DIR = Path(__file__).resolve().parent
MCP_SERVER_FILE = PROJECT_DIR / "MCP_Server.py"


# 配置 MCP 客户端通过 stdio 启动本机 MCP Server。
# 使用 sys.executable，确保子进程使用当前项目的 Python 环境。
mcp_client = MultiServerMCPClient(
    {
        "paper_search": {
            "command": sys.executable,
            "args": [str(MCP_SERVER_FILE)],
            "transport": "stdio",
        }
    }
)

# Agent 的系统提示词，负责约束它如何使用检索工具和回答问题。
AGENT_SYSTEM_PROMPT = """
你是一名严谨的论文阅读助手。你可以使用 search_papers 工具检索本地论文。

工作规则：
1. 用户询问论文内容、研究方法、实验、结论、优缺点等论文相关问题时，必须先调用 search_papers。
2. 调用工具前，结合对话历史理解当前问题；如果用户使用“它”“这个方法”等指代，请将问题整理成独立、明确的检索问题。
3. 本地论文是英文的。传给检索工具的问题应使用简洁、明确的英文，以提高检索效果。
4. 历史对话只用于理解用户问题，不能作为论文事实或证据。
5. 论文事实只能依据工具返回的论文片段，不得编造或用常识补充。
6. 检索片段不足以支持回答时，明确说明“当前检索到的论文内容不足以回答这个问题”。
7. 回答使用用户当前问题的语言；用户用中文提问时用中文回答。
8. 回答时先直接回答，再简要说明依据；尽可能保留检索结果中的论文文件名作为来源。
"""

#不用本地agent 改为mcp tool
# def search_papers(question: str, history: str) -> str:
#     """
#     结合会话历史改写当前问题，然后使用向量检索和关键词检索查找论文片段。
#     调用时把用户当前问题传给 question，把完整历史记录传给 history。
#     """
#     return retrieve_context_with_history({
#         "question": question,
#         "history": history,
#     })

# Agent 可用的工具列表。当前先提供论文混合检索工具。
# agent_tools = [search_papers]
#
# # 创建 Agent。（可以加中间件）
# # Agent 会根据系统提示词决定何时调用 search_papers，以及如何使用检索结果。
# agent = create_agent(
#     model=qa_model,
#     tools=agent_tools,
#     system_prompt=AGENT_SYSTEM_PROMPT,
# )

# 延迟初始化：第一次收到聊天请求时加载 MCP 工具并创建 Agent。
# 后续请求复用同一个 Agent，避免每次请求都重新加载工具。
# _ 是 Python 约定，表示内部/私有变量
_agent = None#存储创建好的agent对象
_agent_tools = None#存储从MCP Server 拿到的工具列表

async def get_agent():
    """加载 MCP 工具，并创建或返回已创建的 Agent。"""
    global _agent, _agent_tools

    if _agent is None:
        # 从 MCP Server 获取工具，并转换成 LangChain 可调用的工具
        _agent_tools = await mcp_client.get_tools()

        if not _agent_tools:
            raise RuntimeError("没有从 MCP_Server.py 加载到任何工具。")

        # 把 MCP 工具交给 Agent。
        # Agent 里不再另外定义重复的 @tool search_papers。
        _agent = create_agent(
            model=qa_model,
            tools=_agent_tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

    return _agent

#内部使用函数
def _message_content_to_text(content) -> str:
    """把模型消息的 content 转成可供 FastAPI 返回的字符串。"""
    # 常见情况：content 本身就是字符串。
    if isinstance(content, str):
        return content

    # 有些模型会把 content 表示成多个文本块组成的列表。
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                text_parts.append(block["text"])

        return "\n".join(text_parts).strip()

    return str(content)


async def ask_agent(question: str, history: str = "") -> str:
    """
    FastAPI 调用这个函数来获取 Agent 的回答。
    参数：
        question：用户当前的问题
        history：当前会话的历史记录
    返回：
        Agent 最终回答的文本
    """
    # 把历史和当前问题一起交给 Agent。
    # 这样 Agent 在调用检索工具前，就能理解“它”“这个方法”等上下文指代。
    agent = await get_agent()
    user_message = f"""
    以下是本次会话的历史记录，仅用于理解当前问题，不是论文证据：
    
    {history.strip() if history.strip() else "（没有历史记录）"}
    
    用户当前的问题：
    {question}
    """

    # create_agent 的输入使用 messages。
    result = await agent.ainvoke({
        "messages": [
            {"role": "user", "content": user_message}
        ]
    })

    # Agent 的结果会包含用户消息、工具调用/返回消息和最终 AI 回答。
    # 从后往前找 最后一条AIMessage
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            answer = _message_content_to_text(message.content)
            if answer:
                return answer

    return "Agent 没有返回回答。"











