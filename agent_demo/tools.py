#!/usr/bin/env python3
"""
Agent Tools - 智能客服工具集
================================
封装RAG和业务API为Agent可调用的标准工具
使用 ChromaDB 持久化向量存储
"""

import os
import sys
from pathlib import Path

# 导入向量存储模块
from vector_store import get_vector_store, ChromaVectorStore


# ============================================================
# Tool 1: 知识库检索 (RAG with ChromaDB)
# ============================================================

class KnowledgeBaseTool:
    """
    知识库检索工具 - 封装 RAG Pipeline
    
    使用 ChromaDB 持久化向量存储：
    - 首次运行自动索引知识库
    - 后续运行直接使用已有索引
    - 支持大规模知识库
    """
    name = "knowledge_search"
    description = "查询微信支付官方政策、费用标准、操作指南等知识。输入：用户问题"
    
    def __init__(self, 
                 knowledge_dir: str = "../rag_demo/knowledge_base",
                 chroma_dir: str = "./chroma_db",
                 force_reindex: bool = False):
        """
        Args:
            knowledge_dir: 知识库文档目录
            chroma_dir: ChromaDB 持久化目录
            force_reindex: 强制重新索引
        """
        self.knowledge_dir = Path(knowledge_dir)
        
        # 使用 ChromaDB 向量存储
        self.vector_store = get_vector_store(
            use_chroma=True,
            persist_directory=chroma_dir,
            collection_name="wxpay_knowledge"
        )
        
        # 检查是否需要索引
        if force_reindex or self.vector_store.count() == 0:
            self._index_documents()
        else:
            print(f"📚 [KnowledgeTool] 使用已有索引 ({self.vector_store.count()} 个文档块)")
    
    def _index_documents(self):
        """加载并索引知识库"""
        if not self.knowledge_dir.exists():
            print(f"⚠️ 知识库目录不存在: {self.knowledge_dir}")
            return
        
        documents = []
        for file_path in self.knowledge_dir.glob("*.txt"):
            print(f"📄 [KnowledgeTool] 加载: {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())
        
        # 智能分块：按章节和段落分割
        chunks = self._smart_chunk(documents)
        
        print(f"🔧 [KnowledgeTool] 正在索引 {len(chunks)} 个文档块到 ChromaDB...")
        self.vector_store.add_documents(chunks)
        print("✅ [KnowledgeTool] 索引完成!")
    
    def _smart_chunk(self, documents: list[str], chunk_size: int = 800) -> list[str]:
        """
        智能分块：按章节标题分割，保持语义完整性
        
        Args:
            documents: 文档列表
            chunk_size: 最大块大小
        """
        chunks = []
        
        for doc in documents:
            lines = doc.split("\n")
            current_chunk = ""
            current_section = ""
            
            for line in lines:
                # 检测章节标题
                if line.startswith("## ") or line.startswith("### "):
                    # 保存之前的块
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    # 开始新块，包含章节标题
                    current_section = line
                    current_chunk = line + "\n"
                elif len(current_chunk) + len(line) < chunk_size:
                    current_chunk += line + "\n"
                else:
                    # 当前块满了，保存并开始新块
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    # 新块以章节标题开头（保持上下文）
                    if current_section:
                        current_chunk = current_section + "\n" + line + "\n"
                    else:
                        current_chunk = line + "\n"
            
            # 保存最后一个块
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        return chunks
    
    def run(self, query: str) -> str:
        """执行知识库检索"""
        results = self.vector_store.search(query, top_k=3)
        
        if not results:
            return "未找到相关知识。"
        
        # 拼接检索结果
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            # 截取关键部分，避免返回太长
            doc_preview = doc[:500] + "..." if len(doc) > 500 else doc
            context_parts.append(f"【{i}】(相关度:{score:.2f})\n{doc_preview}")
        
        context = "\n---\n".join(context_parts)
        return f"【检索到的知识】\n{context}"
    
    def reindex(self):
        """手动触发重新索引"""
        print("🔄 [KnowledgeTool] 清空并重新索引...")
        self.vector_store.clear()
        self._index_documents()


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
