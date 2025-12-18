#!/usr/bin/env python3
"""
RAG实战Demo - 微信支付智能客服
=================================
这是一个完整可运行的RAG示例，模拟微信支付客服场景。

运行方式：
1. 使用OpenAI API（需要API Key）
   set OPENAI_API_KEY=sk-xxx
   python rag_demo.py

2. 使用本地模拟模式（无需API Key，用于理解流程）
   python rag_demo.py --mock

作者：面试准备用Demo
"""

import os
import sys
import argparse
from pathlib import Path

# ============================================================
# 第一部分：文档加载与分块
# ============================================================

def load_documents(knowledge_dir: str) -> list[str]:
    """
    加载知识库文档
    
    Args:
        knowledge_dir: 知识库目录路径
    
    Returns:
        文档内容列表
    """
    documents = []
    knowledge_path = Path(knowledge_dir)
    
    for file_path in knowledge_path.glob("*.txt"):
        print(f"📄 加载文档: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append(f.read())
    
    return documents


def chunk_documents(documents: list[str], chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    使用 LangChain 的 RecursiveCharacterTextSplitter 进行智能分块
    适合中文环境（优先按段落、句子、标点分隔）
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        print("🔪 使用 RecursiveCharacterTextSplitter 进行分块...")
        return text_splitter.split_text("\n\n".join(documents))
    except ImportError:
        print("⚠️ 未找到 langchain，降级使用简易分块...")
        # Fallback implementation
        chunks = []
        for doc in documents:
            current_chunk = ""
            for line in doc.split("\n"):
                if len(current_chunk) + len(line) < chunk_size:
                    current_chunk += line + "\n"
                else:
                    chunks.append(current_chunk)
                    current_chunk = line + "\n"
            if current_chunk: chunks.append(current_chunk)
        return chunks

# ============================================================
# 第二部分：Embedding（向量化）
# ============================================================

class MockEmbedding:
    """
    模拟Embedding类（用于无API Key时演示流程）
    使用简单的词频统计模拟向量
    """
    def __init__(self):
        self.vocab = {}
        self.dim = 100
    
    def embed(self, text: str) -> list[float]:
        """简单的词袋模型模拟Embedding"""
        import hashlib
        # 用hash模拟向量（仅用于演示，实际不能这样做）
        hash_bytes = hashlib.md5(text.encode()).digest()
        return [b / 255.0 for b in hash_bytes[:self.dim]]


class OpenAIEmbedding:
    """
    OpenAI Embedding封装
    """
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
    
    def embed(self, text: str) -> list[float]:
        """调用OpenAI Embedding API"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

# ============================================================
# 第三部分：向量存储与检索
# ============================================================

class SimpleVectorStore:
    """
    简单向量存储（支持 向量检索 和 关键词检索）
    """
    def __init__(self):
        self.documents = []  # 原始文本
        self.embeddings = []  # 对应的向量
    
    def add(self, text: str, embedding: list[float]):
        """添加文档和向量"""
        self.documents.append(text)
        self.embeddings.append(embedding)
    
    def search(self, query: str, query_embedding: list[float], top_k: int = 3) -> list[tuple[str, float]]:
        """
        检索策略：混合检索 (Vector + Keyword)
        如果是 Real Embedding (bge-m3)，向量相似度权重高
        如果是 Mock/Fail，关键词权重高
        """
        import numpy as np
        
        scores = []
        query_tokens = set(query.lower())
        
        # 向量检索算法 (Cosine Similarity)
        use_vector = True
        try:
             # 如果向量维度很小或者全是0，或者是Mock的长度(100)，可能质量不高
             # bge-m3 维度通常是 1024
             if len(query_embedding) < 128 or all(x == 0 for x in query_embedding):
                 use_vector = False
        except:
            use_vector = False

        for i, doc_text in enumerate(self.documents):
            # 1. 关键词得分 (Jaccard)
            doc_tokens = set(doc_text.lower())
            intersection = query_tokens.intersection(doc_tokens)
            union = query_tokens.union(doc_tokens)
            jaccard_score = len(intersection) / len(union) if union else 0
            
            # 特定关键词加强
            keyword_hits = 0
            if "T+1" in query and "T+1" in doc_text: keyword_hits += 1.0
            
            # 2. 向量得分
            vector_score = 0.0
            if use_vector:
                doc_vec = np.array(self.embeddings[i])
                q_vec = np.array(query_embedding)
                norm_q = np.linalg.norm(q_vec)
                norm_d = np.linalg.norm(doc_vec)
                if norm_q > 0 and norm_d > 0:
                    vector_score = np.dot(q_vec, doc_vec) / (norm_q * norm_d)
            
            # 综合打分策略
            if use_vector:
                # 真实Embedding场景：70% 向量 + 30% 关键词 (加强鲁棒性)
                final_score = vector_score * 0.7 + jaccard_score * 0.3 + keyword_hits * 0.2
            else:
                # Mock/Fallback场景：纯关键词
                final_score = jaccard_score * 0.8 + keyword_hits * 0.5
            
            scores.append((i, final_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[i], s) for i, s in scores[:top_k]]

# ============================================================
# 第四部分：LLM生成
# ============================================================

class MockLLM:
    """模拟LLM（用于无API Key时演示）"""
    def generate(self, prompt: str) -> str:
        return f"[模拟回答] 根据提供的资料，我来回答您的问题。（这是模拟输出，实际会调用LLM API）"


class OpenAILLM:
    """OpenAI LLM封装"""
    def __init__(self, model: str = "gpt-3.5-turbo"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
    
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content


class OllamaEmbedding:
    """Ollama Embedding封装"""
    def __init__(self, model: str = "bge-m3"):
        import requests
        self.model = model
        self.base_url = "http://localhost:11434"
    
    def embed(self, text: str) -> list[float]:
        """调用Ollama Embedding API - 失败则返回Mock向量"""
        import requests
        import hashlib
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                emb = response.json().get("embedding")
                if emb: return emb
            else:
                print(f"⚠️ Embedding API Error: {response.text}")
        except Exception as e:
            print(f"⚠️ Embedding Exception: {e}")
        
        # Fallback to Mock (MD5) if API fails
        # This works with our new Keyword Search logic in SimpleVectorStore
        hash_bytes = hashlib.md5(text.encode()).digest()
        # Create a dummy 1024-dim vector to be safe/compatible if needed, but SimpleVectorStore handles short vectors as Mock
        return [b / 255.0 for b in hash_bytes[:100]]


class OllamaLLM:
    """Ollama LLM封装"""
    def __init__(self, model: str = "qwen3:4b"):
        self.model = model
        self.base_url = "http://localhost:11434"
    
    def generate(self, prompt: str) -> str:
        """调用Ollama Chat API (流式版)"""
        import requests
        import json
        
        # 发起流式请求
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,  # 开启流式以获取思考过程
                "options": {
                    "num_ctx": 4096,
                    "num_predict": 2048,  # 调大以便容纳完整的思考+回答
                    "temperature": 0.6,
                }
            },
            stream=True
        )
        
        full_response = ""
        print(f"\n{'='*20} 模型思考与回答 {'='*20}\n")
        
        for line in response.iter_lines():
            if line:
                try:
                    json_obj = json.loads(line.decode('utf-8'))
                    chunk = json_obj.get("response", "")
                    if chunk:
                        print(chunk, end="", flush=True)
                        full_response += chunk
                except:
                    continue
                    
        print(f"\n\n{'='*55}\n")
        return full_response

# ============================================================
# 第五部分：RAG Pipeline
# ============================================================

class RAGPipeline:
    # ... (init and index_documents remain same) ...
    def __init__(self, embedding_model, llm, vector_store):
        self.embedding = embedding_model
        self.llm = llm
        self.vector_store = vector_store
    
    def index_documents(self, chunks: list[str]):
        """索引文档（离线阶段）"""
        print("\n🔧 开始索引文档...")
        for i, chunk in enumerate(chunks):
            emb = self.embedding.embed(chunk)
            self.vector_store.add(chunk, emb)
            if (i + 1) % 5 == 0:
                print(f"   已索引 {i + 1}/{len(chunks)} 个块")
        print("✅ 索引完成!")

    def query(self, question: str, top_k: int = 3) -> str:
        """
        回答问题（在线阶段）
        """
        print(f"\n❓ 问题: {question}")
        
        # Step 1: 问题向量化
        print("   📍 Step 1: 问题Embedding...")
        query_emb = self.embedding.embed(question)
        
        # Step 2: 检索相关文档
        print("   📍 Step 2: 检索相关文档 (混合检索)...")
        # 修改 search 签名，传入 question 文本以便进行关键词检索
        results = self.vector_store.search(question, query_emb, top_k=top_k)
        
        print(f"   📍 检索到 {len(results)} 个相关文档块:")
        for i, (doc, score) in enumerate(results):
            preview = doc[:50].replace("\n", " ") + "..."
            print(f"      [{i+1}] 得分={score:.3f} | {preview}")
        
        # Step 3: 构建Prompt
        print("   📍 Step 3: 构建Prompt (完整上下文)...")
        # 移除 [:500] 截断！允许全部检索结果进入 Prompt
        context = "\n---\n".join([doc for doc, _ in results])
        
        prompt = f"""根据以下参考资料回答问题。如果资料中没有提到，请说不知道。

【参考资料】
{context}

【用户问题】
{question}

【回答】"""
        
        # Step 4: LLM生成
        print("   📍 Step 4: LLM生成回答...")
        answer = self.llm.generate(prompt)
        
        return answer

# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RAG Demo - 微信支付智能客服")
    parser.add_argument("--mock", action="store_true", help="使用模拟模式（无需API Key）")
    parser.add_argument("--ollama", action="store_true", help="使用Ollama本地模型")
    parser.add_argument("--model", type=str, default="qwen3:4b", help="Ollama模型名称")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 RAG实战Demo - 微信支付智能客服")
    print("=" * 60)
    
    # 确定知识库路径
    script_dir = Path(__file__).parent
    knowledge_dir = script_dir / "knowledge_base"
    
    if not knowledge_dir.exists():
        print(f"❌ 知识库目录不存在: {knowledge_dir}")
        sys.exit(1)
    
    # 初始化组件
    if args.ollama:
        print(f"\n⚡ 运行模式: Ollama本地模型 ({args.model})")
        # Ollama 的 embedding 可能不支持所有模型，用 Mock 替代
        embedding = MockEmbedding()
        llm = OllamaLLM(model=args.model)
    elif args.mock:
        print("\n⚡ 运行模式: 模拟模式（不调用真实API）")
        embedding = MockEmbedding()
        llm = MockLLM()
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("\n⚠️  警告: 未设置OPENAI_API_KEY，切换到模拟模式")
            print("   如需使用真实API，请运行: set OPENAI_API_KEY=sk-xxx")
            embedding = MockEmbedding()
            llm = MockLLM()
        else:
            print("\n⚡ 运行模式: OpenAI API模式")
            embedding = OpenAIEmbedding()
            llm = OpenAILLM()
    
    vector_store = SimpleVectorStore()
    
    # 创建RAG Pipeline
    rag = RAGPipeline(embedding, llm, vector_store)
    
    # ===== 离线阶段：索引文档 =====
    print("\n" + "=" * 60)
    print("📚 离线阶段：加载和索引知识库")
    print("=" * 60)
    
    documents = load_documents(str(knowledge_dir))
    chunks = chunk_documents(documents)
    rag.index_documents(chunks)
    
    # ===== 在线阶段：问答 =====
    print("\n" + "=" * 60)
    print("💬 在线阶段：智能问答")
    print("=" * 60)
    
    # 预设问题演示
    demo_questions = [
        "微信支付的结算周期是多久？",
        "商户费率是多少？",
        "如何成为微信支付服务商？",
    ]
    
    for q in demo_questions:
        answer = rag.query(q)
        print(f"\n💡 回答:\n{answer}")
        print("\n" + "-" * 60)
    
    # 交互式问答
    print("\n" + "=" * 60)
    print("🎤 进入交互模式（输入 quit 退出）")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n👤 请输入问题: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 再见！")
                break
            if not user_input:
                continue
            
            answer = rag.query(user_input)
            print(f"\n💡 回答:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break


if __name__ == "__main__":
    main()
