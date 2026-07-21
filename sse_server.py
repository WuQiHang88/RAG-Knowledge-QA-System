import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from sse_starlette.sse import EventSourceResponse
from rank_bm25 import BM25Okapi
import jieba
import json

app = FastAPI()

# ========== 对话记忆配置 ==========
from typing import Dict, List
from langchain_core.messages import BaseMessage, AIMessage

# 短期记忆：会话历史存储（内存，服务重启丢失）
session_history: Dict[str, List[BaseMessage]] = {}
MAX_HISTORY_ROUNDS = 5

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 1. 初始化大模型 ==========
llm = ChatOllama(
    model="deepseek-r1:14b",
    base_url="http://localhost:11434",
    temperature=0.7,
)

# ========== 2. 加载向量数据库（知识库） ==========
embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434",
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="rag_knowledge",
)

# ========== 2.5 新增：长期记忆向量库（独立collection，持久化） ==========
memory_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="user_memory",
)
print(f"长期记忆库加载完成，当前共有 {memory_store._collection.count()} 条记忆")


# ========== 3. 构建 BM25 关键词索引（仅针对知识库） ==========
all_data = vectorstore._collection.get(include=["documents"])
all_docs = all_data["documents"]
all_ids = all_data["ids"]

tokenized_corpus = [list(jieba.cut(doc)) for doc in all_docs]
bm25 = BM25Okapi(tokenized_corpus)

print(f"已加载 {len(all_docs)} 条文档，BM25 索引构建完成")

# ========== 3.5 加载 Reranker 重排序模型 ==========
from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("BAAI/bge-reranker-base", local_files_only=True)
print("Reranker 模型加载完成")


# ========== 4. 混合检索函数（知识库：向量 + BM25） ==========
def hybrid_search(query: str, top_k: int = 15):
    """混合检索：向量 + BM25 关键词，合并去重"""
    vector_results = vectorstore.similarity_search(query, k=top_k)
    vector_texts = [doc.page_content for doc in vector_results]

    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_texts = [all_docs[i] for i in top_indices]

    seen = set()
    merged = []
    for text in vector_texts:
        if text not in seen:
            seen.add(text)
            merged.append(text)
    for text in bm25_texts:
        if text not in seen:
            seen.add(text)
            merged.append(text)

    return merged[:top_k]


# ========== 4.2 新增：长期记忆检索函数 ==========
# 优化后的记忆检索（新增session过滤）
def search_long_term_memory(query: str, session_id: str, top_k: int = 5):
    """仅检索当前会话专属的长期记忆，避免跨会话信息串扰"""
    results = memory_store.similarity_search(
        query,
        k=top_k,
        filter={"session_id": session_id} # 强制按会话隔离记忆
    )
    return [doc.page_content for doc in results]

# ========== 4.3 新增：合并知识库 + 长期记忆的总检索函数 ==========
def hybrid_search_with_memory(query: str, session_id: str, top_k: int = 20):
    kb_docs = hybrid_search(query, top_k=15)
    # 传入当前会话ID，只搜本会话的历史记忆
    memory_docs = search_long_term_memory(query, session_id, top_k=5)

    seen = set()
    merged = []
    for text in kb_docs:
        if text not in seen:
            seen.add(text)
            merged.append(text)
    for text in memory_docs:
        if text not in seen:
            seen.add(text)
            merged.append(text)
    return merged[:top_k]


# ========== 4.5 重排序函数 ==========
def rerank(query: str, docs: list, top_k: int = 3):
    """用 Reranker 对粗排结果精排，返回 top_k"""
    pairs = [[query, doc] for doc in docs]
    scores = reranker_model.predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]


# ========== 服务启动时一次性加载所有文档和元数据 ==========
all_collection_data = vectorstore._collection.get(include=["documents", "metadatas"])
all_docs_text = all_collection_data["documents"]
all_metas = all_collection_data["metadatas"]

doc_to_meta = {}
for i, text in enumerate(all_docs_text):
    if i < len(all_metas):
        doc_to_meta[text] = all_metas[i]


# ========== 4.7 新增：保存长期记忆函数 ==========
def save_to_long_term_memory(question: str, answer: str, session_id: str):
    """把本轮问答存入长期记忆向量库，持久化保存"""
    # 去掉思考过程，只存正式回答（避免<think>标签干扰检索）
    clean_answer = answer
    if "<think>" in clean_answer:
        clean_answer = clean_answer.split("</think>")[-1].strip()

    memory_text = f"""历史对话记录
用户问：{question}
助手答：{clean_answer}"""

    memory_store.add_texts(
        texts=[memory_text],
        metadatas=[{"session_id": session_id, "type": "conversation"}]
    )



# ========== 5. RAG 流式生成（短期+长期记忆双记忆） ==========
async def generate_stream(question: str, session_id: str = "default"):
    # 第一阶段：知识库 + 长期记忆 联合粗排，捞 20 条
    raw_docs = hybrid_search_with_memory(question, session_id, top_k=20)
    # 第二阶段：Reranker 精排，取最相关的 3 条
    docs = rerank(question, raw_docs, top_k=3)

    # 参考来源（知识库显示文件名，记忆库显示「历史对话」）
    sources = []
    for doc_text in docs:
        meta = doc_to_meta.get(doc_text)
        if meta:
            # 知识库文档
            source_path = meta.get("source", "未知")
            filename = source_path.split("\\")[-1].split("/")[-1]
            sources.append({
                "filename": filename,
                "page": meta.get("page", "?")
            })
        else:
            # 长期记忆文档
            sources.append({
                "filename": "历史对话",
                "page": "-"
            })
    sources = sources[:1]

    yield json.dumps({"type": "sources", "sources": sources}, ensure_ascii=False)

    context = "\n\n---\n\n".join(docs)

    # 系统提示词（规则）
    system_prompt = """你是一个知识库问答助手。
【绝对刚性规则，违反即出错】
1. 回答只能使用本次给到的【参考资料】（分为PDF知识库内容、当前会话历史对话记录两类）里的原文信息，严禁使用自身固有知识编造；所有资料都无相关内容时，必须固定回复「知识库与本次会话历史中均无相关内容」，禁止任何形式的猜测扩写。
2. 问题出现「刚才、之前、上一问、那个人、前文提到的」这类会话指代词时，**优先参考标注为【历史对话记录】的当前会话长期记忆**，只有历史记忆完全无匹配内容时，再查阅PDF知识库。
3. 你内部自行校验每条信息对应的来源用于合规校验，**禁止在回答正文里输出任何文件名、页码、历史对话标记、参考资料列表**，不需要向用户展示溯源相关文字。
4. 结合当前短期会话上下文锁定指代的人物/对象，不能跨会话、跨文档随意匹配无关主体。
5. 严格贴合参考资料原文措辞，不得擅自补充年龄、经历、参数这类原文没有的细节。
"""

    # 当前轮用户消息（带参考资料）
    current_user_content = f"""请根据以下【参考资料】回答我的问题。

【参考资料】
{context}

【我的问题】
{question}"""

    # 短期记忆：取出该会话的最近 N 轮历史对话
    history = session_history.get(session_id, [])
    history = history[-(MAX_HISTORY_ROUNDS * 2):]

    # 组装完整消息链：系统提示 + 短期记忆历史 + 当前问题
    messages = [
        SystemMessage(content=system_prompt),
        *history,
        HumanMessage(content=current_user_content),
    ]

    # 流式生成，同时收集完整回答
    full_answer = ""
    async for chunk in llm.astream(messages):
        if chunk.content:
            full_answer += chunk.content
            yield json.dumps({"type": "content", "content": chunk.content}, ensure_ascii=False)

    # ===== 记忆保存 =====
    # 1. 存入短期记忆（内存，会话级）
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=full_answer))
    history = history[-(MAX_HISTORY_ROUNDS * 2):]
    session_history[session_id] = history

    # 2. 存入长期记忆（向量库，持久化）
    save_to_long_term_memory(question, full_answer, session_id)

    yield json.dumps({"type": "done", "done": True}, ensure_ascii=False)


@app.get("/chat/stream")
async def chat_stream(question: str, session_id: str = "default"):
    return EventSourceResponse(generate_stream(question, session_id))


@app.delete("/clear_history")
async def clear_history(session_id: str = "default"):
    """清空指定会话的短期记忆（长期记忆保留）"""
    if session_id in session_history:
        del session_history[session_id]
        return {"status": "ok", "message": f"会话 {session_id} 短期记忆已清空，长期记忆保留"}
    return {"status": "ok", "message": "无短期记忆记录"}


@app.delete("/clear_long_term_memory")
async def clear_long_term_memory():
    """清空全部长期记忆（所有会话的历史对话记忆）"""
    count = memory_store._collection.count()
    if count > 0:
        all_ids = memory_store._collection.get()["ids"]
        memory_store.delete(all_ids)
        return {"status": "ok", "message": f"已清空 {count} 条长期记忆"}
    return {"status": "ok", "message": "暂无长期记忆"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)