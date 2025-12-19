#!/usr/bin/env python3
"""
Vector Store - 向量存储模块
================================
提供持久化的向量存储能力，使用 ChromaDB 实现
支持大规模知识库的高效检索
"""

import os
from pathlib import Path
from typing import Optional, List
import hashlib

# ChromaDB 类型导入
try:
    from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    # 定义占位类型
    EmbeddingFunction = object
    Documents = List[str]
    Embeddings = List[List[float]]


class OllamaEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB 兼容的 Ollama Embedding 函数
    
    继承自 chromadb.api.types.EmbeddingFunction
    """
    
    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def __call__(self, input: Documents) -> Embeddings:
        """批量生成 embedding"""
        import requests
        
        embeddings = []
        for text in input:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=60
                )
                if response.status_code == 200:
                    emb = response.json().get("embedding")
                    if emb:
                        embeddings.append(emb)
                        continue
            except Exception as e:
                print(f"⚠️ Embedding Error: {e}")
            
            # Fallback: 使用 hash 生成固定维度的伪向量
            hash_bytes = hashlib.md5(text.encode()).digest()
            fallback_emb = [b / 255.0 for b in hash_bytes] * 64  # 1024 维
            embeddings.append(fallback_emb[:1024])
        
        return embeddings



class ChromaVectorStore:
    """
    ChromaDB 向量存储
    
    特性:
    - 持久化存储到磁盘
    - 支持大规模文档
    - 高效向量检索
    - 自动去重
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "wxpay_knowledge",
                 embedding_model: str = "bge-m3"):
        """
        Args:
            persist_directory: ChromaDB 持久化目录
            collection_name: 集合名称
            embedding_model: Ollama embedding 模型名
        """
        import chromadb
        from chromadb.config import Settings
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # 创建持久化客户端
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 创建 embedding 函数
        self.embedding_fn = OllamaEmbeddingFunction(model=embedding_model)
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "微信支付智能客服知识库"}
        )
        
        print(f"📦 ChromaDB 已初始化: {self.persist_directory}")
        print(f"   集合: {collection_name}, 现有文档: {self.collection.count()}")
    
    def add_documents(self, documents: list[str], ids: Optional[list[str]] = None):
        """
        添加文档到向量库
        
        Args:
            documents: 文档列表
            ids: 可选的文档ID列表，默认使用内容hash
        """
        if not documents:
            return
        
        # 生成 ID（使用内容 hash 自动去重）
        if ids is None:
            ids = [hashlib.md5(doc.encode()).hexdigest() for doc in documents]
        
        # 检查已存在的 ID，避免重复添加
        existing_ids = set()
        try:
            existing = self.collection.get(ids=ids)
            existing_ids = set(existing.get("ids", []))
        except:
            pass
        
        # 过滤掉已存在的文档
        new_docs = []
        new_ids = []
        for doc, id in zip(documents, ids):
            if id not in existing_ids:
                new_docs.append(doc)
                new_ids.append(id)
        
        if new_docs:
            print(f"📝 添加 {len(new_docs)} 个新文档...")
            self.collection.add(
                documents=new_docs,
                ids=new_ids
            )
            print(f"✅ 添加完成，总文档数: {self.collection.count()}")
        else:
            print(f"ℹ️ 所有文档已存在，跳过添加")
    
    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            [(文档内容, 相似度分数), ...]
        """
        if self.collection.count() == 0:
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )
        
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        # 将距离转换为相似度分数 (距离越小越相似)
        # ChromaDB 默认使用 L2 距离，转换为 0-1 的相似度
        results_with_scores = []
        for doc, dist in zip(documents, distances):
            # 简单的距离到相似度转换
            similarity = 1 / (1 + dist)
            results_with_scores.append((doc, similarity))
        
        return results_with_scores
    
    def clear(self):
        """清空集合"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_fn
        )
        print("🗑️ 集合已清空")
    
    def count(self) -> int:
        """返回文档数量"""
        return self.collection.count()


# 保留简单版本作为 fallback
class SimpleVectorStore:
    """简单向量存储 (内存版，用于回退)"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.embedding_fn = OllamaEmbeddingFunction()
    
    def add_documents(self, documents: list[str], ids: Optional[list[str]] = None):
        """添加文档"""
        for doc in documents:
            if doc not in self.documents:
                self.documents.append(doc)
                emb = self.embedding_fn([doc])[0]
                self.embeddings.append(emb)
    
    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """检索文档"""
        import numpy as np
        
        if not self.documents:
            return []
        
        query_emb = self.embedding_fn([query])[0]
        q_vec = np.array(query_emb)
        
        scores = []
        for i, doc_emb in enumerate(self.embeddings):
            doc_vec = np.array(doc_emb)
            # 余弦相似度
            norm_q = np.linalg.norm(q_vec)
            norm_d = np.linalg.norm(doc_vec)
            if norm_q > 0 and norm_d > 0:
                similarity = np.dot(q_vec, doc_vec) / (norm_q * norm_d)
            else:
                similarity = 0
            scores.append((i, similarity))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[i], s) for i, s in scores[:top_k]]
    
    def count(self) -> int:
        return len(self.documents)


def get_vector_store(use_chroma: bool = True, **kwargs):
    """
    工厂函数：获取向量存储实例
    
    Args:
        use_chroma: 是否使用 ChromaDB（默认 True）
        **kwargs: 传递给向量存储的参数
    """
    if use_chroma:
        try:
            return ChromaVectorStore(**kwargs)
        except Exception as e:
            print(f"⚠️ ChromaDB 初始化失败，回退到内存存储: {e}")
            return SimpleVectorStore()
    else:
        return SimpleVectorStore()
