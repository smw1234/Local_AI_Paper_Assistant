"""
从前端上传pdf，加入本地文件夹并且转化为向量存入向量数据库
"""
from hashlib import sha256
from pathlib import Path

from Vector_Database import db, DOCS_DIR, build_database

def add_pdf_to_database(filename: str, file_bytes: bytes) -> dict:
    """检查重复、保存新 PDF，并将它加入现有向量数据库。"""

    filename = Path(filename).name

    if not filename.lower().endswith(".pdf"):
        raise ValueError("只支持上传 PDF 文件。")

    if not file_bytes:
        raise ValueError("上传的文件为空。")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    new_file_hash = sha256(file_bytes).hexdigest()
    # 检查文件夹中的 PDF：文件名相同，或文件内容相同，都跳过
    for existing_path in DOCS_DIR.iterdir():
        if not existing_path.is_file() or existing_path.suffix.lower() != ".pdf":
            continue

        if existing_path.name.casefold() == filename.casefold():
            return {
                "status": "duplicate",
                "message": f"文件夹中已经有同名论文：{existing_path.name}",
                "filename": existing_path.name,
            }

        existing_hash = sha256(existing_path.read_bytes()).hexdigest()
        if existing_hash == new_file_hash:
            return {
                "status": "duplicate",
                "message": f"文件夹中已经有内容相同的论文：{existing_path.name}",
                "filename": existing_path.name,
            }
    # 保存新论文
    pdf_path = DOCS_DIR / filename
    pdf_path.write_bytes(file_bytes)
    #加入向量库
    build_database()
    return  {
        "message": f"{filename} 已加入向量库"
    }


