#!/usr/bin/env python3
"""医学文献 RAG 模块。

基于 Chroma 向量数据库实现医学文献的加载、分块、索引与检索。
支持智谱（ZhipuAI）Embedding 和 OpenAI 兼容 Embedding。
"""

from __future__ import annotations

import os
import glob
from typing import Optional

from dotenv import load_dotenv

# 加载 .env（确保在读取环境变量之前执行）
load_dotenv()

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# ---------- 常量 ----------
# 默认文献目录（项目根目录下的 medical_docs/）
DEFAULT_DOCS_DIR = os.path.join(os.path.dirname(__file__), "medical_docs")
# Chroma 持久化目录
DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
# 文本分块参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# ---------- 模块级缓存 ----------
_vector_store_cache: Optional[Chroma] = None


def _get_embedding_model() -> OpenAIEmbeddings:
    """获取 Embedding 模型实例。

    优先读取以下环境变量（支持智谱/通义/硅基等国内 API）:
        EMBEDDING_API_KEY   — Embedding 服务的 API Key
        EMBEDDING_BASE_URL  — Embedding 服务的 Base URL
        EMBEDDING_MODEL     — Embedding 模型名称

    如果未设置 EMBEDDING_* 变量，则回退到 OPENAI_* 变量。

    Returns:
        OpenAIEmbeddings 实例。

    Raises:
        ValueError: 缺少 API Key 配置时抛出。
    """
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv(
        "OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"
    )
    model = os.getenv("EMBEDDING_MODEL", "embedding-3")

    if not api_key:
        raise ValueError(
            "未配置 Embedding API Key。请在 .env 文件中设置 EMBEDDING_API_KEY "
            "或 OPENAI_API_KEY 环境变量。"
        )

    print(f"[RAG] Embedding 配置: model={model}, base_url={base_url}")

    return OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )


def load_documents(docs_dir: str = DEFAULT_DOCS_DIR) -> list[Document]:
    """从指定目录加载文献文档。

    支持的格式: .txt, .md, .pdf

    Args:
        docs_dir: 文献目录路径。

    Returns:
        加载的 Document 列表。
    """
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"文献目录不存在: {docs_dir}")

    documents: list[Document] = []

    # 加载 .txt 文件
    txt_files = glob.glob(os.path.join(docs_dir, "**", "*.txt"), recursive=True)
    for fpath in txt_files:
        try:
            loader = TextLoader(fpath, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            print(f"[警告] 加载 {fpath} 失败: {e}")

    # 加载 .md 文件（使用 TextLoader 避免依赖 unstructured）
    md_files = glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True)
    for fpath in md_files:
        try:
            loader = TextLoader(fpath, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            print(f"[警告] 加载 {fpath} 失败: {e}")

    # 加载 .pdf 文件
    pdf_files = glob.glob(os.path.join(docs_dir, "**", "*.pdf"), recursive=True)
    for fpath in pdf_files:
        try:
            loader = PyPDFLoader(fpath)
            documents.extend(loader.load())
        except Exception as e:
            print(f"[警告] 加载 {fpath} 失败: {e}")

    if not documents:
        raise ValueError(f"在 {docs_dir} 中未找到任何文献文件（支持 .txt/.md/.pdf）")

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """将文档分块。

    Args:
        documents: 原始文档列表。
        chunk_size: 每块最大字符数。
        chunk_overlap: 块之间的重叠字符数。

    Returns:
        分块后的 Document 列表。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(
    docs_dir: str = DEFAULT_DOCS_DIR,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    force_rebuild: bool = False,
) -> Chroma:
    """构建或加载 Chroma 向量数据库。

    使用模块级缓存，避免重复加载。如果 persist_dir 已存在且
    force_rebuild=False，则直接加载已有数据库。否则从文献目录重新构建。

    Args:
        docs_dir: 文献目录路径。
        persist_dir: Chroma 持久化目录。
        force_rebuild: 是否强制重建。

    Returns:
        Chroma 向量数据库实例。
    """
    global _vector_store_cache

    # 使用缓存（非强制重建时）
    if _vector_store_cache is not None and not force_rebuild:
        return _vector_store_cache

    embedding = _get_embedding_model()

    # 尝试加载已有数据库
    if os.path.isdir(persist_dir) and not force_rebuild:
        try:
            db = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding,
            )
            count = db._collection.count()
            if count > 0:
                print(f"[RAG] 已加载向量数据库，共 {count} 个文档块")
                _vector_store_cache = db
                return db
        except Exception as e:
            print(f"[RAG] 加载已有数据库失败: {e}，将重新构建...")

    # 重新构建
    print(f"[RAG] 从 {docs_dir} 加载文献...")
    documents = load_documents(docs_dir)
    print(f"[RAG] 共加载 {len(documents)} 个文档")

    chunks = split_documents(documents)
    print(f"[RAG] 文档分块完成，共 {len(chunks)} 个块")

    print("[RAG] 正在构建向量索引（调用 Embedding API）...")
    try:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=persist_dir,
        )
    except Exception as e:
        raise RuntimeError(
            f"Embedding API 调用失败: {e}\n"
            f"请检查 .env 中的配置：\n"
            f"  EMBEDDING_API_KEY={os.getenv('EMBEDDING_API_KEY', '(未设置)')[:8]}...\n"
            f"  EMBEDDING_BASE_URL={os.getenv('EMBEDDING_BASE_URL', '(未设置)')}\n"
            f"  EMBEDDING_MODEL={os.getenv('EMBEDDING_MODEL', '(未设置)')}"
        ) from e

    print(f"[RAG] 向量数据库构建完成，已持久化至 {persist_dir}")
    _vector_store_cache = db
    return db


def search(
    query: str,
    top_k: int = 5,
    docs_dir: str = DEFAULT_DOCS_DIR,
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> list[dict]:
    """检索与查询最相关的文献片段。

    Args:
        query: 查询文本。
        top_k: 返回的最相关结果数。
        docs_dir: 文献目录路径。
        persist_dir: Chroma 持久化目录。

    Returns:
        检索结果列表，每项包含 content, source, score。
    """
    db = build_vector_store(docs_dir=docs_dir, persist_dir=persist_dir)

    results = db.similarity_search_with_relevance_scores(query, k=top_k)

    output = []
    for doc, score in results:
        output.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "未知来源"),
            "score": round(score, 4),
        })

    return output


def format_search_results(results: list[dict]) -> str:
    """将检索结果格式化为可读文本。

    Args:
        results: search() 返回的结果列表。

    Returns:
        格式化的文本。
    """
    if not results:
        return "未找到相关文献。"

    parts = []
    for i, r in enumerate(results, 1):
        source_name = os.path.basename(r["source"])
        parts.append(
            f"【结果 {i}】相关度: {r['score']}\n"
            f"来源: {source_name}\n"
            f"内容: {r['content']}\n"
        )
    return "\n".join(parts)


# ---------- CLI 入口：单独运行时可用于构建索引 ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="医学文献 RAG - 索引构建与检索")
    parser.add_argument("--build", action="store_true", help="构建/重建向量索引")
    parser.add_argument("--query", type=str, help="检索查询")
    parser.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR, help="文献目录")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数")
    args = parser.parse_args()

    if args.build:
        build_vector_store(docs_dir=args.docs_dir, force_rebuild=True)
    elif args.query:
        results = search(args.query, top_k=args.top_k, docs_dir=args.docs_dir)
        print(format_search_results(results))
    else:
        parser.print_help()
