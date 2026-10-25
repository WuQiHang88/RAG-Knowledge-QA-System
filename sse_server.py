import os
# 国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import json
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from sse_starlette.sse import EventSourceResponse
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import CrossEncoder
from typing import Dict, List
from langchain_core.messages import BaseMessage, AIMessage

app = FastAPI()

# ========== 对话记忆配置 ==========
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

# ========== 2.5 长期记忆向量库 ==========
memory_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="user_memory",
)
print(f"长期记忆库加载完成，当前共有 {memory_store._collection.count()} 条记忆")

# ========== 3. SPLADE 稀疏语义检索模块 ==========
# 替换原有的BM25
# 中文场景推荐替换为中文SPLADE模型，如 shibing624/splade-bert-base-chinese；这个是V3版本，naver/splade-v3；这个是V2版本最强的那个，naver/splade-cocondenser-ensembledistil
SPLADE_MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
SPLADE_MAX_LENGTH = 512
SPLADE_BATCH_SIZE = 8

print("正在加载 SPLADE 稀疏检索模型...")
splade_tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL_NAME)
splade_model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL_NAME)
splade_model.eval()
if torch.cuda.is_available():
    splade_model = splade_model.cuda()
    print("SPLADE 模型已加载至 GPU")


def splade_encode(texts: List[str]) -> List[dict]:
    """用 SPLADE 模型批量生成稀疏向量（词权重字典）"""
    inputs = splade_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=SPLADE_MAX_LENGTH,
        return_tensors="pt"
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        output = splade_model(**inputs)

    # SPLADE 标准计算：log(1 + relu(logits)) + max pooling
    scores = torch.log1p(torch.relu(output.logits))
    scores = torch.max(scores, dim=1).values  # [batch_size, vocab_size]

    # 转为稀疏字典，过滤零权重
    sparse_vectors = []
    for i in range(scores.shape[0]):
        nonzero_idx = torch.nonzero(scores[i]).squeeze(1)
        nonzero_val = scores[i][nonzero_idx]
        vec_dict = {int(idx): float(val) for idx, val in zip(nonzero_idx, nonzero_val)}
        sparse_vectors.append(vec_dict)
    return sparse_vectors


def sparse_dot_product(query_vec: dict, doc_vec: dict) -> float:
    """计算两个稀疏向量的点积相似度"""
    score = 0.0
    for token_id, weight in query_vec.items():
        score += weight * doc_vec.get(token_id, 0.0)
    return score


# 服务启动时构建全量稀疏索引
print("正在构建文档 SPLADE 稀疏索引...")
all_collection_data = vectorstore._collection.get(include=["documents", "metadatas"])
all_docs = all_collection_data["documents"]
all_metas = all_collection_data["metadatas"]

# 批量生成稀疏向量，避免显存溢出
all_sparse_vectors = []
for i in range(0, len(all_docs), SPLADE_BATCH_SIZE):
    batch = all_docs[i:i + SPLADE_BATCH_SIZE]
    all_sparse_vectors.extend(splade_encode(batch))

print(f"SPLADE 索引构建完成，共 {len(all_docs)} 条文档")

# ========== 3.5 加载 Reranker 重排序模型 ==========
reranker_model = CrossEncoder("BAAI/bge-reranker-base", local_files_only=True)
print("Reranker 模型加载完成")

# 构建文本到元数据的映射
doc_to_meta = {}
for i, text in enumerate(all_docs):
    if i < len(all_metas):
        doc_to_meta[text] = all_metas[i]


# ========== 4. 混合检索函数（稠密向量 + SPLADE 稀疏语义） ==========
def hybrid_search(query: str, top_k: int = 15):
    """混合检索：稠密向量深层语义 + SPLADE稀疏语义，合并去重"""
    # 第一路：稠密向量检索（bge-m3）
    vector_results = vectorstore.similarity_search(query, k=top_k)
    vector_texts = [doc.page_content for doc in vector_results]

    # 第二路：SPLADE 稀疏语义检索
    query_sparse = splade_encode([query])[0]
    scores = [sparse_dot_product(query_sparse, doc_vec) for doc_vec in all_sparse_vectors]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    splade_texts = [all_docs[i] for i in top_indices]

    # 合并两路结果，去重
    seen = set()
    merged = []
    for text in vector_texts:
        if text not in seen:
            seen.add(text)
            merged.append(text)
    for text in splade_texts:
        if text not in seen:
            seen.add(text)
            merged.append(text)

    return merged[:top_k]


# ========== 4.2 长期记忆检索函数 ==========
def search_long_term_memory(query: str, session_id: str, top_k: int = 5):
    results = memory_store.similarity_search(
        query,
        k=top_k,
        filter={"session_id": session_id}
    )
    return [doc.page_content for doc in results]


# ========== 4.3 知识库 + 长期记忆 联合检索 ==========
def hybrid_search_with_memory(query: str, session_id: str, top_k: int = 20):
    kb_docs = hybrid_search(query, top_k=15)
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
    pairs = [[query, doc] for doc in docs]
    scores = reranker_model.predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]


# ========== 4.7 保存长期记忆函数 ==========
def save_to_long_term_memory(question: str, answer: str, session_id: str):
    clean_answer = answer
    if "" in clean_answer:
        clean_answer = clean_answer.split("")[-1].strip()
    memory_text = f"""历史对话记录
用户问：{question}
助手答：{clean_answer}"""
    memory_store.add_texts(
        texts=[memory_text],
        metadatas=[{"session_id": session_id, "type": "conversation"}]
    )


# ========== 5. RAG 流式生成 ==========
async def generate_stream(question: str, session_id: str = "default"):
    raw_docs = hybrid_search_with_memory(question, session_id, top_k=20)
    docs = rerank(question, raw_docs, top_k=3)

    sources = []
    for doc_text in docs:
        meta = doc_to_meta.get(doc_text)
        if meta:
            source_path = meta.get("source", "未知")
            filename = source_path.split("\\")[-1].split("/")[-1]
            sources.append({
                "filename": filename,
                "page": meta.get("page", "?")
            })
        else:
            sources.append({
                "filename": "历史对话",
                "page": "-"
            })
    sources = sources[:1]
    yield json.dumps({"type": "sources", "sources": sources}, ensure_ascii=False)

    context = "\n\n---\n\n".join(docs)

    system_prompt = """你是一个知识库问答助手。
【绝对刚性规则，违反即出错】
1. 回答只能使用本次给到的【参考资料】（分为PDF知识库内容、当前会话历史对话记录两类）里的原文信息，严禁使用自身固有知识编造；所有资料都无相关内容时，必须固定回复「知识库与本次会话历史中均无相关内容」，禁止任何形式的猜测扩写。
2. 问题出现「刚才、之前、上一问、那个人、前文提到的」这类会话指代词时，**优先参考标注为【历史对话记录】的当前会话长期记忆**，只有历史记忆完全无匹配内容时，再查阅PDF知识库。
3. 你内部自行校验每条信息对应的来源用于合规校验，**禁止在回答正文里输出任何文件名、页码、历史对话标记、参考资料列表**，不需要向用户展示溯源相关文字。
4. 结合当前短期会话上下文锁定指代的人物/对象，不能跨会话、跨文档随意匹配无关主体。
5. 严格贴合参考资料原文措辞，不得擅自补充年龄、经历、参数这类原文没有的细节。
"""

    current_user_content = f"""请根据以下【参考资料】回答我的问题。
【参考资料】
{context}
【我的问题】
{question}"""

    history = session_history.get(session_id, [])
    history = history[-(MAX_HISTORY_ROUNDS * 2):]

    messages = [
        SystemMessage(content=system_prompt),
        *history,
        HumanMessage(content=current_user_content),
    ]

    full_answer = ""
    async for chunk in llm.astream(messages):
        if chunk.content:
            full_answer += chunk.content
            yield json.dumps({"type": "content", "content": chunk.content}, ensure_ascii=False)

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=full_answer))
    history = history[-(MAX_HISTORY_ROUNDS * 2):]
    session_history[session_id] = history

    save_to_long_term_memory(question, full_answer, session_id)

    yield json.dumps({"type": "done", "done": True}, ensure_ascii=False)


@app.get("/chat/stream")
async def chat_stream(question: str, session_id: str = "default"):
    return EventSourceResponse(generate_stream(question, session_id))


@app.delete("/clear_history")
async def clear_history(session_id: str = "default"):
    if session_id in session_history:
        del session_history[session_id]
        return {"status": "ok", "message": f"会话 {session_id} 短期记忆已清空，长期记忆保留"}
    return {"status": "ok", "message": "无短期记忆记录"}


@app.delete("/clear_long_term_memory")
async def clear_long_term_memory():
    count = memory_store._collection.count()
    if count > 0:
        all_ids = memory_store._collection.get()["ids"]
        memory_store.delete(all_ids)
        return {"status": "ok", "message": f"已清空 {count} 条长期记忆"}
    return {"status": "ok", "message": "暂无长期记忆"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
