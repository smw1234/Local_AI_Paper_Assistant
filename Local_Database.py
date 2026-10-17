from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
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
for pdf_path in DOCS_DIR.glob("*.pdf"):
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    # print(f"\n--- {pdf_path.name} ---")
    # print(text[:1000])  # 先看前 1000 个字符

    #切分成文本块
    chunks=splitter.split_text(text)
    # for i, chunk in enumerate(chunks):
    #     print(f"\nchunk{i}:",chunk)

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
       )#转化为向量
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

#5.做成可用工具



