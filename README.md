\# RAG 知识库问答系统

基于 FastAPI + LangChain + Ollama 搭建私有化RAG问答平台

\## 技术栈

FastAPI、LangChain、Chroma、BM25、BGE-Reranker、SSE流式通信、Python、Markdown、Ollama、向量检索、长/短记忆

\## 项目亮点

1\. BM25+向量混合检索 + Reranker精排，提升文档检索准确率

2\. 长短双持久化记忆机制，区分「清空对话」与「清空全部记忆」

3\. SSE流式输出，前端Markdown实时渲染，支持参考来源溯源

4\. 强约束提示词，降低大模型幻觉

\## 快速启动

1\. 创建虚拟环境，安装依赖

`pip install -r requirements.txt`

2\. 准备PDF文档，放入提前建好的“docs”文件夹，执行ingest.py向量化入库

3\. 启动后端：`python sse\_server.py`

4\. 打开demo.html访问问答页面

