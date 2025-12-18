#!/usr/bin/env python3
"""
Agent Tools - 智能客服工具集
================================
封装RAG和业务API为Agent可调用的标准工具
"""

import os
import sys
from pathlib import Path

# ============================================================
# 基础类 (从 rag_demo 复用)
# ============================================================

class OllamaEmbedding:
    """Ollama Embedding封装 (bge-m3)"""
    def __init__(self, model: str = "bge-m3"):
        self.model = model
        self.base_url = "http://localhost:11434"
    
    def embed(self, text: str) -> list[float]:
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
        except Exception as e:
            print(f"⚠️ Embedding Error: {e}")
        
        # Fallback
        hash_bytes = hashlib.md5(text.encode()).digest()
        return [b / 255.0 for b in hash_bytes[:100]]


class SimpleVectorStore:
    """简单向量存储 (混合检索)"""
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add(self, text: str, embedding: list[float]):
        self.documents.append(text)
        self.embeddings.append(embedding)
    
    def search(self, query: str, query_embedding: list[float], top_k: int = 3) -> list[tuple[str, float]]:
        import numpy as np
        scores = []
        query_tokens = set(query.lower())
        
        use_vector = len(query_embedding) >= 128
        
        for i, doc_text in enumerate(self.documents):
            doc_tokens = set(doc_text.lower())
            intersection = query_tokens.intersection(doc_tokens)
            union = query_tokens.union(doc_tokens)
            jaccard_score = len(intersection) / len(union) if union else 0
            
            vector_score = 0.0
            if use_vector:
                doc_vec = np.array(self.embeddings[i])
                q_vec = np.array(query_embedding)
                norm_q = np.linalg.norm(q_vec)
                norm_d = np.linalg.norm(doc_vec)
                if norm_q > 0 and norm_d > 0:
                    vector_score = np.dot(q_vec, doc_vec) / (norm_q * norm_d)
            
            if use_vector:
                final_score = vector_score * 0.7 + jaccard_score * 0.3
            else:
                final_score = jaccard_score
            
            scores.append((i, final_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[i], s) for i, s in scores[:top_k]]


# ============================================================
# Tool 1: 知识库检索 (RAG)
# ============================================================

class KnowledgeBaseTool:
    """
    知识库检索工具 - 封装 RAG Pipeline
    
    用于回答：费率、结算周期、政策规则等静态知识问题
    """
    name = "knowledge_search"
    description = "查询微信支付官方政策、费用标准、操作指南等知识。输入：用户问题"
    
    def __init__(self, knowledge_dir: str = "../rag_demo/knowledge_base"):
        self.embedding = OllamaEmbedding()
        self.vector_store = SimpleVectorStore()
        self._index_documents(knowledge_dir)
    
    def _index_documents(self, knowledge_dir: str):
        """加载并索引知识库"""
        knowledge_path = Path(knowledge_dir)
        if not knowledge_path.exists():
            print(f"⚠️ 知识库目录不存在: {knowledge_dir}")
            return
        
        documents = []
        for file_path in knowledge_path.glob("*.txt"):
            print(f"📄 [KnowledgeTool] 加载: {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())
        
        # 简单分块
        chunks = []
        for doc in documents:
            current = ""
            for line in doc.split("\n"):
                if len(current) + len(line) < 500:
                    current += line + "\n"
                else:
                    if current.strip(): chunks.append(current.strip())
                    current = line + "\n"
            if current.strip(): chunks.append(current.strip())
        
        print(f"🔧 [KnowledgeTool] 索引 {len(chunks)} 个文档块...")
        for chunk in chunks:
            emb = self.embedding.embed(chunk)
            self.vector_store.add(chunk, emb)
        print("✅ [KnowledgeTool] 索引完成!")
    
    def run(self, query: str) -> str:
        """执行知识库检索"""
        query_emb = self.embedding.embed(query)
        results = self.vector_store.search(query, query_emb, top_k=3)
        
        if not results:
            return "未找到相关知识。"
        
        # 拼接检索结果
        context = "\n---\n".join([doc for doc, _ in results])
        return f"【检索到的知识】\n{context}"


# ============================================================
# Tool 2: 订单查询 (模拟业务API)
# ============================================================

class OrderQueryTool:
    """
    订单查询工具 - 模拟业务系统API
    
    用于回答：订单状态、退款进度等动态业务问题
    """
    name = "order_query"
    description = "查询订单/退款状态。输入：订单号或退款单号 (如 ORDER_1001, REF_999)"
    
    # 模拟订单数据库
    MOCK_DB = {
        "ORDER_1001": {"status": "已完成", "amount": 99.00, "time": "2024-12-15 14:30", "refund": None},
        "ORDER_1002": {"status": "退款中", "amount": 199.00, "time": "2024-12-10 09:00", "refund": "REF_2001"},
        "ORDER_1003": {"status": "待支付", "amount": 59.00, "time": "2024-12-18 10:00", "refund": None},
        "REF_2001": {"status": "处理中", "original_order": "ORDER_1002", "amount": 199.00, "eta": "1-3个工作日"},
        "REF_2002": {"status": "已退款", "original_order": "ORDER_999", "amount": 50.00, "completed": "2024-12-17"},
    }
    
    def run(self, order_id: str) -> str:
        """查询订单/退款状态"""
        order_id = order_id.strip().upper()
        
        # 如果用户只输入了数字，尝试自动补全前缀
        if order_id.isdigit():
            # 尝试匹配 ORDER_ 或 REF_ 前缀
            possible_ids = [f"ORDER_{order_id}", f"REF_{order_id}"]
            for pid in possible_ids:
                if pid in self.MOCK_DB:
                    order_id = pid
                    break
        
        if order_id in self.MOCK_DB:
            record = self.MOCK_DB[order_id]
            if order_id.startswith("REF"):
                return f"【退款单 {order_id}】状态: {record['status']}, 原订单: {record['original_order']}, 金额: ¥{record['amount']}"
            else:
                refund_info = f", 关联退款: {record['refund']}" if record['refund'] else ""
                return f"【订单 {order_id}】状态: {record['status']}, 金额: ¥{record['amount']}, 下单时间: {record['time']}{refund_info}"
        else:
            return f"未找到订单号 {order_id}，请核对后重新输入。"


# ============================================================
# 工具注册表
# ============================================================

def get_all_tools() -> dict:
    """返回所有可用工具"""
    return {
        "knowledge_search": KnowledgeBaseTool(),
        "order_query": OrderQueryTool(),
    }
