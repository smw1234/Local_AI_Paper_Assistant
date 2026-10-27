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
import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

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
           batch_vectors=[item.embedding for item in response_data]#response.data是一个列表
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

from FlagEmbedding import FlagReranker
# 延迟加载：避免每次问答都重新加载模型。
# 第一次实际需要重排时才加载，之后复用。
_reranker = None

#加载高精度二分排位模型 对混合检索的结果进行精选
def get_reranker():
    global _reranker

    if _reranker is None:
        # use_fp16=False：CPU/GPU 都较稳妥，但速度会慢一些。
        # 模型首次加载时需要下载；下载后会缓存到本地。
        _reranker = FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=False,
        )

    return _reranker


def rerank_documents(query, documents, top_n=11):
    """用 Reranker 对 RRF 初筛出的论文片段重新排序。"""
    if not documents:
        return []

    reranker = get_reranker()

    # Reranker 对每个“问题、论文片段”进行组合
    pairs = [
        [query, doc.page_content]
        for doc in documents
    ]
    scores = reranker.compute_score(pairs)#计算相关性分数

    # 只有一个候选片段时，模型可能只返回一个分数而非分数列表。如果抛异常，就将scores转化为列表
    try:
        len(scores)
    except TypeError:
        scores = [scores]

    # 分数越高，表示模型判断该论文片段与问题越相关。ranked里既有分数又有文档
    ranked = sorted(
        zip(scores, documents),#把分数和文档进行配对组成元组
        key=lambda item: item[0],#item是列表里的每一个元素，即元组 取出每个元组里的分数进行降序排序
        reverse=True,#降序排列
    )
    #取前八个元组，只返回文档不返回分数
    return [doc for score, doc in ranked[:top_n]]

#定义检索器
def retrieve_context(query,k=11,candidate_k=15):
    """向量检索和 BM25 关键词检索，再用 RRF 合并排名。之后用reranker精排"""
    #向量检索
    vector_docs = db.similarity_search_by_vector(
        embed_query(query),#把查询问题转化为向量
        k=candidate_k,
    )
    #关键词检索
    keyword_docs = keyword_search(query, k=candidate_k)
    #准备融合
    fused_scores = {}#记录每个文档的分数
    docs_by_key = {}#记录每个文档块的原文
    #把两个榜单里出现的所有文档块，计算分数存入fused_scores，对应的文档存入docs_by_key
    for ranked_docs in (vector_docs, keyword_docs):#外层循环两次 一次向量列表 一个关键词列表
        for rank, doc in enumerate(ranked_docs, start=1):#序号从1开始，元素先是向量k个后是k个关键词
            #元数据取出
            key = (
                doc.metadata.get("source"),
                doc.metadata.get("chunk_index"),
            )
            #计算分数对应存入 同一个文本块同时出现在两个榜单里分数会累加
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (60 + rank)
            docs_by_key[key] = doc
    #按融合分数排序 对fused_scores进行排序，排序依据是value 取前k
    ranked_keys = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )[:candidate_k]
    #按融合分数排序好的文档对象列表
    candidate_docs = [
        docs_by_key[key]
        for key in ranked_keys
    ]

    if not candidate_docs:
        return "没有检索到相关论文内容。"

    #Reranker 精排，只保留最终需要的 k 个片段
    final_docs = rerank_documents(
        query=query,
        documents=candidate_docs,
        top_n=k,
    )

    return "\n\n".join(
        f"[论文：{doc.metadata.get('source', '未知')}]\n"
        f"{doc.page_content}"
        for doc in final_docs
    )

#将问题结合历史记录进行改写
rewrite_p="""
根据对话历史，把用户当前问题改写成一个独立、明确、适合检索论文内容的问题。
要求：
1. 解决“它”“这个方法”“上述结果”等指代。
2. 保留当前问题的本意，不要回答问题，不要添加历史中没有的信息。
3. 只输出改写后的问题，不要解释。
4.论文是英文的。请将当前问题改写成简洁、明确的英文检索问题。
5.只输出英文检索问题，不要回答。

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


def tokenize(text):
    """把英文文本拆成单个的小写关键词，只保留字母和数字。"""
    #re是正则表达式 r:\不是转义字符 [a-zA-Z0-9]+：任连续一串字母（不分大小写）或数字 +表示连续
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def keyword_search(query, k=11):
    """从 Chroma 取出文本块，用 BM25算法按关键词匹配度进行打分，取分数最高的k个返回。"""
    stored = db._collection.get(
        include=["documents", "metadatas"]
    )
    texts = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    if not texts:
        return []
    #把文本块和问题进行切词
    tokenized_texts = [tokenize(text or "") for text in texts]
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    #构建BM25模型 传入全部预料
    bm25 = BM25Okapi(tokenized_texts)
    #根据问题关键词给所有文档打分 分数越高越相关 返回分数列表
    scores = bm25.get_scores(query_tokens)
    #按分数排序索引 排分数会到底对不上文档
    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,#降序
    )
    #按照索引排名，将前k个最相关的文档和元数据打包返回
    results = []
    for i in ranked_indexes:
        if scores[i] <= 0:
            break
        results.append(
            Document(
                page_content=texts[i],
                metadata=metadatas[i] or {},
            )
        )
        if len(results) >= k:
            break
    return results

import logging

logger = logging.getLogger(__name__)

#带有历史记录的检索
def retrieve_context_with_history(inputs):
     """先结合历史改写问题，再用改写结果检索论文。"""
     question = inputs["question"]
     history = inputs.get("history", "")
     search_query =rewrite_question(question, history)
     # print(f"原问题：{question}")
     # print(f"改写后的检索问题：{search_query}")
     logger.info("原问题：%s", question)
     logger.info("改写后的检索问题：%s", search_query)
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




