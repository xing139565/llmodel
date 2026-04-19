import os
# 设置 HuggingFace 国内镜像源，防止连不上外网导致卡死
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import logging
import pickle
import jieba
from rank_bm25 import BM25Okapi
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 添加跨域资源共享 (CORS) 配置，允许前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 线上环境建议改成具体的域名，如 ["http://localhost:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. embedding模型
embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")

# 2. 向量数据库 (持久化)
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
collection = chroma_client.get_or_create_collection(name="docs")

# 3. BM25 索引 (如果存在)
bm25_model = None
bm25_chunks = []
bm25_metadatas = []
if os.path.exists("./data/bm25_corpus.pkl"):
    try:
        with open("./data/bm25_corpus.pkl", "rb") as f:
            bm25_data = pickle.load(f)
            bm25_model = BM25Okapi(bm25_data["tokenized_corpus"])
            bm25_chunks = bm25_data["chunks"]
            bm25_metadatas = bm25_data["metadatas"]
            logging.info("Successfully loaded BM25 corpus.")
    except Exception as e:
        logging.error(f"Error loading BM25: {e}")

class Query(BaseModel):
    question: str
    history: list = []


# ===== 调用 Ollama =====
def call_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 32768
                }
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        logging.error(f"Error calling Ollama: {e}")
        return "本地模型调用失败，请检查 Ollama 服务是否运行。"


def rewrite_question(history, question):
    if not history:
        return question
        
    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-3:]])
    prompt = f"""
你是一个专业的问题重写助手。
请结合以下用户与助手的对话历史，将用户最新的问题重写为一个独立、完整的提问。
重写后的问题必须包含所有必要的业务上下文（例如主语、限定词等），并且不要包含任何多余的解释。

--- 对话历史 ---
{history_text}

--- 用户最新问题 ---
{question}

请直接输出重写后的问题：
"""
    rewritten = call_ollama(prompt).strip()
    return rewritten if rewritten else question


def route_question_rules(question: str) -> str:
    q = question.lower().strip()

    chitchat_kw = ["你好", "hello", "hi", "在吗", "谢谢", "再见", "你是谁", "你能做什么", "早上好", "晚安"]
    if any(q == k or (k in q and len(q) <= 6) for k in chitchat_kw):
        return "chitchat"

    global_menu_kw = ["所有菜单", "全部菜单", "有哪些菜单", "系统菜单", "全局菜单", "菜单介绍", "菜单结构", "系统功能", "有哪些功能模块"]
    if any(k in q for k in global_menu_kw):
        return "global_menu"
        
    route_kw = ["菜单", "路由", "路径", "入口", "页面在哪里", "哪些页面"]
    if any(k in q for k in route_kw):
        return "route_qa"

    manual_kw = ["怎么", "如何", "步", "操作", "点击", "配置", "新增", "修改", "导出", "详情", "使用"]
    if any(k in q for k in manual_kw):
        return "manual_qa"
        
    req_kw = ["为什么", "需求", "规则", "逻辑", "背景", "方案", "口径", "计算", "业务规则"]
    if any(k in q for k in req_kw):
        return "requirement_qa"

    return "general_rag"


def build_answer_prompt(route_type: str, history_text: str, original_question: str, context: str) -> str:
    base = f"""
你是智慧能源与系统产品的资深专家助手。请严格根据以下知识库检索内容回答问题。
严禁凭借自己的通用大模型常识强行捏造、编造本系统并不存在的业务细节、菜单或功能步骤。

--- 对话历史 ---
{history_text}

--- 知识库检索内容片断 ---
{context}

--- 用户当前原问题 ---
{original_question}
"""

    if route_type == "chitchat":
        return base + "\n【重要指令】：用户仅仅是在打招呼或寒暄，请自然、亲切地回复即可，简单自报家门（智慧能源知识库助手即可），不需刻意说明“知识库没有找到信息”。"
        
    if route_type == "global_menu":
        return base + "\n【重要指令】：请结合上方提取的全部配置，按结构化、目录化的方式汇总梳理出我们的菜单介绍（重点列举菜单层级骨架和路径）。尽量使用层级列表清晰地排版框架！"
        
    if route_type == "route_qa":
        return base + """\n【重要指令】：
这是路由与菜单结构问题。
请优先回答：
1. 菜单上下级归属
2. 明确出现于知识库中的路由路径
3. 页面主要作用
4. 明确出现于知识库中的权限信息

如果知识库片段中没有明确写出路径、权限、入口、上下级关系，
必须直接写“知识库未明确给出”，禁止使用“可能是、通常是、应为、大概率是”等猜测措辞。
"""

    if route_type == "manual_qa":
        return base + "\n【重要指令】：偏向操作指南类。请基于资料，以最高效、清晰的“分为步骤X”的形式给出操作指导，指出需要点击系统的什么模块什么页面、怎么配置，以及相关的注意事项。"

    if route_type == "requirement_qa":
        return base + "\n【重要指令】：偏向需求背景与计算规则类。请为用户详细解释背后的业务原由和计算逻辑的来源，侧重于系统“为什么”这么设计，重点剖析计算方式与口径。"

    return base + "\n【重要指令】：请基于检索提取的证据片段，专业、准确、连贯地为用户给出详实解答。若查找不到文档信息请明确抱歉并说明知识库未录入该点。"

def get_where_filter(route_type: str):
    if route_type == "route_qa":
        return {"category": "routes"}
    if route_type == "manual_qa":
        return {"category": "manuals"}
    if route_type == "requirement_qa":
        return {"category": "requirements"}
    return None

def call_ollama_stream(prompt: str, sources: list, route_type: str):
    # 下发前置的元数据信息
    meta = {
        "type": "meta",
        "sources": sources,
        "route_type": route_type
    }
    yield json.dumps(meta, ensure_ascii=False) + "\n"

    try:
        with requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_ctx": 32768
                }
            },
            stream=True,
            timeout=180
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    chunk_text = data.get("response", "")
                    if chunk_text:
                        yield json.dumps({"type": "chunk", "content": chunk_text}, ensure_ascii=False) + "\n"
    except Exception as e:
        yield json.dumps({"type": "chunk", "content": f"\n\n[连接大模型中途发生截断: {str(e)}]"}, ensure_ascii=False) + "\n"

def retrieve_for_global_query():
    # 针对全局菜单问题触发兜底逻辑：绕过向量搜寻，直接拉取数据库里标注为 routes 的结构数据
    data = collection.get(where={"category": "routes"})
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    
    if not docs:
        return "未能提取到全局配置或菜单信息", []
        
    matched_parts = []
    sources = []
    for i, doc in enumerate(docs):
        source = metas[i].get("source", "未知配置") if i < len(metas) else "未知配置"
        matched_parts.append(f"【来源：{source}】\n{doc}")
        sources.append(source)
    
    # 防止溢出上下文窗口，取前150个核心块（通过拓展到更大的上下文窗口处理全量数据）
    matched_parts = matched_parts[:150]
    return "\n\n".join(matched_parts), list(set(sources))


# ===== 问答接口 =====
@app.post("/chat")
def chat(query: Query):
    raw_question = query.question
    history = query.history
    
    # 获取初步路由意图
    route_type = route_question_rules(raw_question)
    logging.info(f"Soft Router Detected Intention: {route_type}")

    # 问题重写
    search_query = rewrite_question(history, raw_question)
    logging.info(f"Original Query: {raw_question} | Rewritten Search Query: {search_query}")

    context = ""
    sources = []

    # 寒暄跳过检索，全局菜单直接兜底，两者以外走Reranker深度精排
    if route_type == "chitchat":
        context = "闲聊无需知识库内容支持"
    elif route_type == "global_menu":
        logging.info("Triggered Global Menu Bypass!")
        context, sources = retrieve_for_global_query()
    else:
        where_filter = get_where_filter(route_type)
        
        # 1. 向量检索 (ChromaDB)
        query_embedding = embedding_model.encode(search_query).tolist()
        chroma_kwargs = {"query_embeddings": [query_embedding], "n_results": 60}
        if where_filter:
            chroma_kwargs["where"] = where_filter
            
        results = collection.query(**chroma_kwargs)
        chroma_docs = results.get("documents", [[]])[0]
        chroma_metas = results.get("metadatas", [[]])[0]

        if not chroma_docs and where_filter:
            # 过滤条件无结果时降级兜底
            results = collection.query(query_embeddings=[query_embedding], n_results=60)
            chroma_docs = results.get("documents", [[]])[0]
            chroma_metas = results.get("metadatas", [[]])[0]

        # 2. 关键词检索 (BM25)
        bm25_docs = []
        bm25_meta_list = []
        if bm25_model is not None:
            tokenized_query = jieba.lcut(search_query)
            scores = bm25_model.get_scores(tokenized_query)
            doc_scores = [(idx, score) for idx, score in enumerate(scores)]
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            for idx, score in doc_scores[:60]:
                if score > 0:
                    if where_filter:
                        k = list(where_filter.keys())[0]
                        v = where_filter[k]
                        if bm25_metadatas[idx].get(k) != v:
                            continue
                    bm25_docs.append(bm25_chunks[idx])
                    bm25_meta_list.append(bm25_metadatas[idx])

        # 3. 取并集去重 (Deduplication)
        docs_set = set()
        combined_docs = []
        combined_metas = []
        
        for d, m in zip(chroma_docs + bm25_docs, chroma_metas + bm25_meta_list):
            if d not in docs_set:
                docs_set.add(d)
                combined_docs.append(d)
                combined_metas.append(m)

        if not combined_docs:
            context = "当前检索不到任何相关内容片段。"
        else:
            pairs = [[search_query, doc] for doc in combined_docs]
            scores = reranker.predict(pairs)
            
            scored_results = list(zip(combined_docs, combined_metas, scores))
            scored_results.sort(key=lambda x: x[2], reverse=True)
            
            # 动态阈值截取策略 (扩容 + 及格线)：
            # 1. 强制保底录取前 3 个片段，即使它们得分极低，也避免无材料可用。
            # 2. 录取所有交叉打分 (Logits) > -1.0 的相关片段。
            # 3. 为防止上下文撑爆显存或断崖式拉低速度，硬性截断封顶在 40 个片段。
            top_docs = []
            top_metas = []
            for d, m, s in scored_results:
                if len(top_docs) < 3:
                    top_docs.append(d)
                    top_metas.append(m)
                elif s > -1.0 and len(top_docs) < 40:
                    top_docs.append(d)
                    top_metas.append(m)
                else:
                    if len(top_docs) >= 40 or s <= -1.0:
                        break
            
            context = "\n\n---\n\n".join(top_docs)
            sources = list(set([m.get("source", "未知文件") for m in top_metas]))

    history_text = ""
    if history:
        history_text = "\n".join([f"{h['role']}：{h['content']}" for h in history[-3:]])

    prompt = build_answer_prompt(route_type, history_text, raw_question, context)

    return StreamingResponse(
        call_ollama_stream(prompt, sources, route_type),
        media_type="application/x-ndjson"
    )
