import os

# 国内镜像源，与后端保持一致
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import CrossEncoder
from typing import List

# ========== 1. 加载向量库、嵌入模型 ==========
embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434",
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    collection_name="rag_knowledge",
    embedding_function=embeddings,
)

# ========== 2. SPLADE 稀疏语义检索模块（与 sse_server 完全对齐） ==========
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


# 构建全量文档稀疏索引
print("正在构建文档 SPLADE 稀疏索引...")
all_data = vectorstore._collection.get(include=["documents"])
all_docs = all_data["documents"]
all_ids = all_data["ids"]

# 批量生成稀疏向量，避免显存溢出
all_sparse_vectors = []
for i in range(0, len(all_docs), SPLADE_BATCH_SIZE):
    batch = all_docs[i:i + SPLADE_BATCH_SIZE]
    all_sparse_vectors.extend(splade_encode(batch))

print(f"SPLADE 索引构建完成，共 {len(all_docs)} 条文档")

# ========== 3. 加载 Reranker 重排序模型 ==========
reranker_model = CrossEncoder("BAAI/bge-reranker-base")


# ========== 4. 混合检索函数（稠密向量 + SPLADE 稀疏语义） ==========
def hybrid_search(query: str, top_k: int = 20):
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


# ========== 5. Reranker 重排序函数 ==========
def rerank(query: str, docs: list, top_k: int = 3):
    pairs = [[query, doc] for doc in docs]
    scores = reranker_model.predict(pairs)
    scored_docs = list(zip(docs, scores))
    # 按分数从高到低排序
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:top_k]


# ========== 测试用例 ==========
test_cases = [
    ("Transformer模型中，自注意力机制（Self-Attention）的计算过程分为哪三步？Q、K、V分别代表什么？",
     ["自注意力", "Self-Attention", "Query", "Key", "Value"]),
    ("attention优点是什么？",
     ["参数少", "速度快", "效果好"]),
    ("任职过什么公司？",
     []),
    ("什么是Bert",
     ["2018年", "双向模型", "纯文本语料库", "Wikipedia"]),
    ("多邻域任务处理会有什么影响？",
     ["干扰效应", "泛化性", "学习速率"]),
    ("PyTorch是什么？",
     ["Meta", "深度学习", "Torch", "动态计算图"]),
    ("计算图的组成是？",
     ["Tensor", "算子", "节点"]),
    ("动态计算图的主要特点",
     ["即时执行", "灵活性", "易于调试", "研究友好"]),
    ("静态计算图的主要特点",
     ["预先定义", "优化", "部署友好", "可移植性"]),
    ("PyTorch常用模块介绍",
     ["torch", "torch.nn", "torch.utils.data"]),
]


# ========== 测试函数：对比【仅向量top5】 vs 【SPLADE混合粗排+Reranker精排top3】 ==========
def test_recall():
    vec_hit_count = 0  # 仅向量检索top5 命中数
    rerank_hit_count = 0  # SPLADE混合+Reranker精排top3 命中数
    results = []
    for question, keywords in test_cases:
        print(f"\n==================== 测试问题：{question} ====================")
        # 1. 纯向量top5 原始召回
        vec_docs = vectorstore.similarity_search(question, k=5)
        vec_all_text = "\n".join([d.page_content for d in vec_docs])
        vec_hit = all(kw in vec_all_text for kw in keywords)
        if vec_hit:
            vec_hit_count += 1

        # 2. SPLADE混合粗排20条 + Reranker打分重排，取top3
        raw_20_docs = hybrid_search(question, top_k=20)
        rerank_top3 = rerank(question, raw_20_docs, top_k=3)

        # 拼接Reranker选出的3条文本，判断是否命中关键词
        rerank_all_text = "\n".join([doc for doc, score in rerank_top3])
        rerank_hit = all(kw in rerank_all_text for kw in keywords)
        if rerank_hit:
            rerank_hit_count += 1

        # 打印Reranker打分与最终3条片段
        print("【Reranker重排后Top3 相关性打分（分数越高越相关）】")
        for idx, (doc_text, score) in enumerate(rerank_top3, 1):
            preview = doc_text[:120] + "..." if len(doc_text) > 120 else doc_text
            print(f"  Top{idx} | 相关性分数：{score:.4f} | 片段：{preview}")

        # 保存单条用例结果
        results.append({
            "question": question,
            "keywords": keywords,
            "vec_hit": vec_hit,
            "rerank_hit": rerank_hit,
            "vec_top1_preview": vec_docs[0].page_content[:80] + "..." if vec_docs else "无",
            "rerank_top1_score": rerank_top3[0][1] if rerank_top3 else 0,
            "rerank_top1_text": rerank_top3[0][0][:80] + "..." if rerank_top3 else "无",
        })

    # 汇总打印召回率对比
    print("\n" + "=" * 100)
    print(f"召回率对比测试汇总（SPLADE 版本）")
    print(f"总测试用例：{len(test_cases)}")
    print(f"1. 仅向量检索 top5  | 命中数：{vec_hit_count} | 召回率：{vec_hit_count / len(test_cases) * 100:.1f}%")
    print(
        f"2. SPLADE混合检索+Reranker精排 top3 | 命中数：{rerank_hit_count} | 召回率：{rerank_hit_count / len(test_cases) * 100:.1f}%")
    print("=" * 100)

    # 逐条打印详细对比
    print("\n逐条用例详细结果：")
    for i, r in enumerate(results, 1):
        vec_status = "✅ 命中" if r["vec_hit"] else "❌ 未命中"
        rerank_status = "✅ 命中" if r["rerank_hit"] else "❌ 未命中"
        print(f"\n{i}. 问题：{r['question']}")
        print(f"   关键词：{r['keywords']}")
        print(f"   纯向量top5：{vec_status} | top1片段：{r['vec_top1_preview']}")
        print(
            f"   SPLADE混合+Reranker top3：{rerank_status} | top1分数：{r['rerank_top1_score']:.4f} | top1片段：{r['rerank_top1_text']}")

    vec_rate = vec_hit_count / len(test_cases)
    rerank_rate = rerank_hit_count / len(test_cases)
    print(
        f"\n【总结】SPLADE方案召回率变化：{rerank_rate * 100:.1f}%（原纯向量{vec_rate * 100:.1f}%），提升{(rerank_rate - vec_rate) * 100:.1f}%")
    return vec_rate, rerank_rate


if __name__ == "__main__":
    test_recall()
