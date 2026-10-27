from mcp.server.fastmcp import FastMCP
from Vector_Database import retrieve_context_with_history

# 创建 MCP Server
mcp = FastMCP(
  "Thesis Paper Assistant",
        host="127.0.0.1",
        port=8001,

)
#定义工具
@mcp.tool()
def search_papers(question: str, history: str = "") -> str:
    """搜索本地论文。
    Args:
        question：要检索的用户问题。
        history：可选的对话历史，用于改写有上下文的问题。
    Returns:
         返回检索到的论文片段和来源。
    """
    return retrieve_context_with_history({
        "question": question,
        "history": history,
    })


if __name__ == "__main__":
    # 默认使用 stdio，供本机 MCP 客户端启动并连接，不需要 token 或端口。
    #防止工具每次调用时都重启，将stdio改为http
    mcp.run(transport="streamable-http")