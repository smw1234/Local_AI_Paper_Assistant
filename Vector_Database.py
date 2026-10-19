from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_deepseek import ChatDeepSeek
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv(override=True)

#模型初始化
qa_model=ChatDeepSeek(
    model="deepseek-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

client=OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

# 本地 PDF 文件夹
DOCS_DIR = Path(r"D:\DESKTOP\thesis\all_thesis")
#向量数据库文件夹
DB_DIR = Path("C:/Users/Lenovo/PycharmProjects/Local_ai_Paper_Assistant/chroma_db")

# 创建或打开本地 Chroma 数据库
db = Chroma(
    collection_name="papers",
    persist_directory=str(DB_DIR),
    embedding_function=None,#不需要自己将文本转化为向量
)
#1.定义切割器
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=180,
    separators=[
        "\n\n",  # 段落
        "\n",    # 换行
        ". ",    # 英文句号后有空格
        "? ",
        "! ",
        "; ",
        ": ",
        ", ",
        " ",     # 单词边界
        "",      # 按字符拆分
    ]
)
# 2.读取论文并切割转化为向量放入向量数据库
def build_database():
    """
    pdf没入库时 读取 PDF、切分、生成向量，并写入本地 Chroma
    """
    if not DOCS_DIR.exists():
        print(f"论文文件夹不存在：{DOCS_DIR}")
        return
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"没有在 {DOCS_DIR} 中找到 PDF。")
        return

    for pdf_path in pdf_files:
        # 查 Chroma：是否已经有这篇论文的记录
        existing=db._collection.get(
            where={"source": pdf_path.name},
            include=["metadatas"]
        )
        if existing["ids"]:
            print(f"已存在，跳过：{pdf_path.name}")
            continue

        # 只处理尚未入库的论文
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        #切分成文本块
        chunks=splitter.split_text(text)
        ##筛选文本块 去掉空白块
        chunks = [chunk for chunk in chunks if chunk.strip()]

        #建立本地向量索引
        vectors_count = 0 #记录当前文件向量总数
        for start in range(0, len(chunks), 10):#每十个文本块一组 0 10 20
           batch = chunks[start:start+10]#放入batch里
           response=client.embeddings.create(
               model="text-embedding-v3",
               input=batch,
               dimensions=1024,
               encoding_format="float"
           )
           #转化为向量 按字段进行排序
           response_data = sorted(response.data, key=lambda item: item.index)
           batch_vectors=[item.embedding for item in response.data]#response.data是一个列表
           # 为每个文本块记录论文文件名和块序号（主键）
           ids = [
               f"{pdf_path.name}-chunk-{start + i}"
               for i in range(len(batch))
           ]
           #生成元数据(属性）
           metadatas = [
               {
                   "source": pdf_path.name,
                   "chunk_index": start + i,
               }
               for i in range(len(batch))
           ]
           # 写入本地 Chroma
           db._collection.upsert(
               ids=ids,
               documents=batch,#存入文本
               embeddings=batch_vectors,#存入向量
               metadatas=metadatas,
           )
           vectors_count += len(batch_vectors)

        print(f"{pdf_path.name}: {len(chunks)} 个文本块，{vectors_count} 个向量")
    print(f"向量数据库位置：{DB_DIR.resolve()}")

#4.实现问答
def embed_query(text:str):
    """
    手动将用户问题转化为向量
    """
    resp=client.embeddings.create(
        model="text-embedding-v3",
        input=[text],
        dimensions=1024,
        encoding_format="float"
    )
    return resp.data[0].embedding

#定义检索器
def retrieve_context(question):
    docs = db.similarity_search_by_vector(
        embed_query(question),
        k=8,
    )

    if not docs:
        return "没有检索到相关论文内容。"

    return "\n\n".join(
        f"[论文：{doc.metadata.get('source', '未知')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )

rewrite_p="""
根据对话历史，把用户当前问题改写成一个独立、明确、适合检索论文内容的问题。
要求：
1. 解决“它”“这个方法”“上述结果”等指代。
2. 保留当前问题的本意，不要回答问题，不要添加历史中没有的信息。
3. 只输出改写后的问题，不要解释。

对话历史：
{history}

当前问题：
{question}

改写后的检索问题：
"""
def rewrite_question(question, history):
    """结合历史，把当前问题改写成适合检索的独立问题。"""
    if not history.strip():
        return question
    rewrite_prompt = ChatPromptTemplate.from_messages([("human", rewrite_p)])
    result = qa_model.invoke(
        rewrite_prompt.format_messages(
            history=history,
            question=question,
        )
    )

    rewritten = result.content.strip()
    return rewritten or question
            
            

#带有历史记录的检索
def retrieve_context_with_history(inputs):
     """先结合历史改写问题，再用改写结果检索论文。"""
     question = inputs["question"]
     history = inputs.get("history", "")
     search_query =rewrite_question(question, history)
     print(f"原问题：{question}")
     print(f"改写后的检索问题：{search_query}")
     return retrieve_context(search_query)

def make_chain():
    """
    创建检索问答链
    """
    #定义prompt
    messages = """
    你是一名严谨的论文阅读助手。请根据检索到的论文片段回答用户问题。

    规则：
    1. 论文事实只能依据检索到的论文片段，不要编造。
    2. 历史对话只用于理解上下文，不能当作论文证据。
    3. 如果论文片段不足以回答，明确说明。
    4. 回答时先直接回答，再简要说明依据。
    5. 在相关陈述后标注论文来源。
    6. 使用与当前问题相同的语言回答。

    历史对话：
    {history}

    用户当前问题：
    {question}

    检索到的论文片段：
    {context}
    """
    prompt_temp=ChatPromptTemplate.from_messages([("human",messages)])
    #定义链
    chain=  {
            "question": RunnableLambda(lambda inputs: inputs["question"]),
            "history": RunnableLambda(lambda inputs: inputs.get("history", "")),
            "context": RunnableLambda(retrieve_context_with_history)
        } | prompt_temp | qa_model
    return chain

#5.做成可用工具
def chat_loop(chain):
    """循环接收问题，输入 q 或 exit 退出。"""
    print("\n论文助手已启动。输入 q 或 exit 退出。")

    while True:
        question = input("\n请输入问题：").strip()

        if question.lower() in {"q", "exit"}:
            print("已退出。")
            break

        if not question:
            continue

        try:
            result = chain.invoke(question)
            print("\n回答：")
            print(result.content)
        except Exception as e:
            print(f"问答时出错：{e}")

def main():
    # 首次运行或添加了新论文时，调用这个函数建库
    build_database()
    # 创建问答链并进入交互循环
    chain = make_chain()
    chat_loop(chain)


if __name__ == "__main__":
    main()

#检索优化 向量检索+关键词检索
#docker部署



