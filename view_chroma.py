from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 加载向量库
embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434",
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="rag_knowledge",
)

# 获取 collection
collection = vectorstore._collection

# 统计信息
count = collection.count()
print(f"向量库总片段数：{count}")
print("=" * 80)

# 获取所有数据
all_data = collection.get(include=["documents", "metadatas"])

# 按来源文件分组
from collections import defaultdict

file_groups = defaultdict(list)
for i, doc in enumerate(all_data["documents"]):
    source = all_data["metadatas"][i].get("source", "未知")
    page = all_data["metadatas"][i].get("page", "?")
    file_groups[source].append((page, doc))

# 按文件打印统计
print("\n按文件统计：")
for source, chunks in file_groups.items():
    print(f"  📄 {source} → {len(chunks)} 个片段")

# 浏览前 N 个片段
print("\n" + "=" * 80)
N = 5  # 看前5条，改这个数字看多一点
print(f"\n前 {N} 个片段预览：\n")

for i in range(min(N, count)):
    doc = all_data["documents"][i]
    meta = all_data["metadatas"][i]
    source = meta.get("source", "?")
    page = meta.get("page", "?")

    print(f"--- 第 {i + 1} 条 | 文件: {source} | 第 {page} 页 ---")
    print(doc[:200] + "..." if len(doc) > 200 else doc)
    print()

# 搜索测试
print("=" * 80)
query = input("\n输入一个关键词搜索试试（直接回车跳过）：")
if query.strip():
    results = vectorstore.similarity_search(query, k=3)
    print(f"\n搜索「{query}」的 top3 结果：\n")
    for i, doc in enumerate(results, 1):
        print(f"--- 结果 {i} ---")
        print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
        print()