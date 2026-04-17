from fastapi import FastAPI
from pydantic import BaseModel
import requests
from sentence_transformers import SentenceTransformer
import chromadb
import logging

app = FastAPI()

# 1. embedding模型
# 按需求使用小模型
embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 2. 向量数据库 (持久化)
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
collection = chroma_client.get_or_create_collection(name="docs")

class Query(BaseModel):
    question: str


# ===== 调用 Ollama =====
def call_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        logging.error(f"Error calling Ollama: {e}")
        return "本地模型调用失败，请检查 Ollama 服务是否运行。"


# ===== 问答接口 =====
@app.post("/chat")
def chat(query: Query):
    question = query.question

    # 1. 向量检索
    query_embedding = embedding_model.encode(question).tolist()

    # 检索 top 3-5，这里取 3
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        context = "无相关内容"
        sources = []
    else:
        context = "\n\n---\n\n".join(docs)
        # 提取来源文件名并去重
        sources = list(set([m.get("source", "未知文件") for m in metas]))

    # 2. 构造提示词
    # 按照需求指定的提示词
    prompt = f"""
你是公司平台产品助手。
请严格根据提供的知识库内容回答问题。
如果知识库中没有足够信息，请明确回答“未找到相关信息”，不要自行编造。
回答时优先给出明确步骤、功能位置、限制条件和注意事项。

--- 知识库内容 ---
{context}

--- 用户问题 ---
{question}
"""

    answer = call_ollama(prompt)

    return {
        "answer": answer,
        "sources": sources
    }