"""Step 8：RAG —— 检索增强生成（含上下文压缩检索）。

演示最基础的 RAG 流程：
    1. WebBaseLoader 抓网页 → RecursiveCharacterTextSplitter 切块 → Chroma 向量化建库
    2. 基础检索（base_retriever）按相似度召回最相关的文档片段
    3. ContextualCompressionRetriever + LLMChainExtractor 对召回结果「压缩」，
       只保留与问题相关的片段，去掉无关上下文，省 token 也更聚焦
    4. ContextualCompressionRetriever + EmbeddingsFilter 按「向量相似度阈值」过滤，
       不调用 LLM，更便宜更快，适合对召回结果做粗筛

运行：
    python examples/step8_rag.py
"""
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    EmbeddingsFilter,
    LLMChainExtractor,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm import get_model


def pretty_print_docs(docs) -> None:
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


def test():
    loader = WebBaseLoader("https://mp.weixin.qq.com/s/7sZhZPeBE7XmBLjik8al8w")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    splits = text_splitter.split_documents(data)

    llm = get_model()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 基础检索器：向量库 + 相似度检索
    base_retriever = Chroma.from_documents(
        documents=splits, embedding=embeddings
    ).as_retriever()
    question = "这个网页主要是关于哪方面的？"

    print("=== 原始检索结果 ===")
    docs = base_retriever.invoke(question)
    pretty_print_docs(docs)

    # 上下文压缩检索：先按相似度召回，再用 LLM 从每篇文档里「抽取」和问题相关的片段，
    # 丢弃无关部分，最终喂给 LLM 的上下文更精炼、更省 token。
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    print("\n=== 压缩后的检索结果 ===")
    compressed_docs = compression_retriever.invoke(question)
    pretty_print_docs(compressed_docs)

    # 向量相似度过滤：把召回文档的 embedding 和问题的 embedding 算余弦相似度，
    # 低于 similarity_threshold 的文档直接丢弃（不调用 LLM，便宜且快）。
    # k 可进一步限制最多保留几篇；threshold 设太小会保留太多、太大则可能全被过滤。
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.7,  # 相似度阈值，按需调整
        k=3,                        # 最多保留 3 篇
    )
    filter_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=base_retriever,
    )

    print("\n=== 向量相似度过滤后的检索结果 ===")
    filtered_docs = filter_retriever.invoke(question)
    pretty_print_docs(filtered_docs)


if __name__ == "__main__":
    test()
