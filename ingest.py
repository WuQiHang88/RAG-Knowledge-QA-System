import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "./docs"

embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434",
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=180,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)


def clean_text(text: str) -> str:
    """清洗PDF提取的文本：去掉字间空格、多余换行"""
    # 去掉中文字符之间的空格（保留英文单词间的空格）
    # 简单粗暴：所有单字空格合并
    text = re.sub(r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])', r'\1\2', text)
    text = re.sub(r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])', r'\1\2', text)  # 跑两遍确保干净
    # 多余空行压缩
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


# 加载已有向量库
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="rag_knowledge",
)
collection = vectorstore._collection

# 查询已有文件
existing_data = collection.get(include=["metadatas"])
existing_files = set()
for meta in existing_data["metadatas"]:
    source = meta.get("source", "")
    filename = os.path.basename(source)
    existing_files.add(filename)

print(f"向量库已有 {len(existing_files)} 个文件")

# 找出新文件
new_files = []
for filename in os.listdir(DOCS_DIR):
    if filename.lower().endswith(".pdf"):
        if filename not in existing_files:
            new_files.append(filename)

if not new_files:
    print("\n✅ 没有新文件")
    # 建议：删掉 chroma_db 文件夹重建，因为旧的文本没清洗过
    print("⚠️  注意：旧数据没有经过文本清洗，建议删除 chroma_db 后重新全量入库")
    exit()

print(f"新文件：{new_files}")

# 处理新文件
all_splits = []
for filename in new_files:
    filepath = os.path.join(DOCS_DIR, filename)
    print(f"\n处理：{filename}")

    loader = PyPDFLoader(filepath)
    documents = loader.load()

    # 关键：逐页清洗文本
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    splits = text_splitter.split_documents(documents)
    print(f"  切分 {len(splits)} 块")
    all_splits.extend(splits)

# 入库
print(f"\n共 {len(all_splits)} 块，入库中...")
vectorstore.add_documents(all_splits)
print("✅ 完成")