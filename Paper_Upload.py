"""
从前端上传pdf，加入本地文件夹并且转化为向量存入向量数据库
"""
from hashlib import sha256
from pathlib import Path
import json
from Vector_Database import db, DOCS_DIR, build_database
#论文哈希存放地址
HASH_CACHE_PATH = DOCS_DIR / ".paper_hash_cache.json"

def load_hash_cache() -> dict:
    """读取本地哈希缓存；缓存不存在或损坏时返回空字典。"""
    try:
        return json.loads(HASH_CACHE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def get_cached_hash(pdf_path: Path, hash_cache: dict) -> str:
    """文件未变化时直接使用缓存；变化后才重新计算哈希。"""
    stat = pdf_path.stat()
    cached = hash_cache.get(pdf_path.name)

    if (
        cached
        and cached.get("size") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
    ):
        return cached["sha256"]

    file_hash = sha256(pdf_path.read_bytes()).hexdigest()
    hash_cache[pdf_path.name] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_hash,
    }
    return file_hash


def save_hash_cache(hash_cache: dict) -> None:
    """保存哈希缓存。"""
    HASH_CACHE_PATH.write_text(
        json.dumps(hash_cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_pdf_to_database(filename: str, file_bytes: bytes) -> dict:
    """检查重复、保存新 PDF，并将它加入现有向量数据库。"""

    filename = Path(filename).name#只保留最后一段文件名
    #endswith：判断后缀一不一样
    if not filename.lower().endswith(".pdf"):
        raise ValueError("只支持上传 PDF 文件。")

    if not file_bytes:
        raise ValueError("上传的文件为空。")
    #参数：自动创建缺失的父目录 目录已存在时不报错
    DOCS_DIR.mkdir(parents=True, exist_ok=True)#目录不存在就创建
    new_file_hash = sha256(file_bytes).hexdigest()#对文件内容算 SHA-256 哈希，得到 64 位十六进制字符串（hexdigest）

    hash_cache = load_hash_cache()
    duplicate_path = None

    # 检查文件夹中的 PDF：文件名相同，或文件内容相同，都跳过
    for existing_path in DOCS_DIR.iterdir():
        #跳过子目录 or 跳过非pdf文件
        if not existing_path.is_file() or existing_path.suffix.lower() != ".pdf":
            continue
        # #按文件名查重 casefold()：比 lower() 更彻底的大小写不敏感比较
        # if existing_path.name.casefold() == filename.casefold():
        #     return {
        #         "status": "duplicate",
        #         "message": f"文件夹中已经有同名论文：{existing_path.name}",
        #         "filename": existing_path.name,
        #     }

        #按内容哈希查重 对每个已存在的 PDF 实时读取内容再算哈希
        # existing_hash = sha256(existing_path.read_bytes()).hexdigest()

        #优化 只按内容判断是否重复
        existing_hash = get_cached_hash(existing_path, hash_cache)
        if existing_hash == new_file_hash:
            duplicate_path = existing_path
            break
    # 即使发现重复，也要先保存刚刚计算过的哈希缓存
    save_hash_cache(hash_cache)
    if duplicate_path:
        return {
                    "status": "duplicate",
                    "message": f"文件夹中已经有内容相同的论文：{existing_path.name}",
                    "filename": existing_path.name,
                }
    # 保存新论文
    pdf_path = DOCS_DIR / filename #路径拼接

    # 同名但内容不同，给新文件加哈希后缀，避免覆盖
    if pdf_path.exists():
        pdf_path = DOCS_DIR / f"{pdf_path.stem}_{new_file_hash[:8]}{pdf_path.suffix}"

    pdf_path.write_bytes(file_bytes)#写入磁盘

    stat = pdf_path.stat()
    hash_cache[pdf_path.name] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": new_file_hash,
    }
    save_hash_cache(hash_cache)
    #加入向量库
    build_database()
    return  {
        "message": f"{filename} 已加入向量库"
    }
#优化：建库时把哈希缓存，不用每次都计算对比

