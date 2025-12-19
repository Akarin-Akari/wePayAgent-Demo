#!/usr/bin/env python3
"""
Memory System - 对话记忆模块
================================
为 ReActAgent 提供短期记忆和长期记忆能力

短期记忆 (ConversationMemory): 保存当前会话的对话历史
长期记忆 (SummaryMemory): 对话摘要 + 实体记忆，支持持久化
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import hashlib


class ConversationMemory:
    """
    短期记忆 - 当前会话对话历史
    
    特性：
    - 保留最近 N 轮对话
    - 超出时自动丢弃最旧的对话
    - 支持清空重置
    """
    
    def __init__(self, max_turns: int = 10):
        """
        Args:
            max_turns: 最大保留的对话轮数 (一轮 = 用户输入 + 助手回复)
        """
        self.max_turns = max_turns
        self.history: list[dict] = []
    
    def add(self, user_message: str, assistant_message: str):
        """添加一轮对话"""
        self.history.append({
            "role": "user",
            "content": user_message
        })
        self.history.append({
            "role": "assistant", 
            "content": assistant_message
        })
        
        # 超出上限时，移除最早的对话轮
        while len(self.history) > self.max_turns * 2:
            self.history.pop(0)  # 移除最早的 user
            self.history.pop(0)  # 移除最早的 assistant
    
    def get_context(self) -> list[dict]:
        """获取对话历史作为 LLM 上下文"""
        return self.history.copy()
    
    def get_last_n_turns(self, n: int = 3) -> list[dict]:
        """获取最近 N 轮对话"""
        return self.history[-(n * 2):] if self.history else []
    
    def clear(self):
        """清空短期记忆"""
        self.history = []
    
    def __len__(self):
        return len(self.history) // 2  # 返回对话轮数


class SummaryMemory:
    """
    长期记忆 - 对话摘要 + 实体记忆
    
    特性:
    - 存储对话摘要
    - 提取并记住关键实体 (订单号、退款单号等)
    - JSON 持久化存储
    """
    
    def __init__(self, storage_path: str = "./memory_store"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.summaries: list[dict] = []  # 历史会话摘要
        self.entities: dict = {}  # 实体记忆 {entity_type: {entity_id: info}}
        
        # 实体提取模式 (简单正则匹配)
        self.entity_patterns = {
            "order_id": r"ORDER_\d+",
            "refund_id": r"REF_\d+",
        }
    
    def extract_entities(self, text: str) -> dict:
        """从文本中提取实体"""
        import re
        extracted = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                extracted[entity_type] = list(set(matches))
        return extracted
    
    def remember_entities(self, text: str):
        """记住文本中提到的实体"""
        entities = self.extract_entities(text)
        for entity_type, ids in entities.items():
            if entity_type not in self.entities:
                self.entities[entity_type] = {}
            for eid in ids:
                self.entities[entity_type][eid] = {
                    "first_mentioned": datetime.now().isoformat(),
                    "last_mentioned": datetime.now().isoformat()
                }
    
    def get_known_entities(self) -> str:
        """获取已知实体的文本描述"""
        if not self.entities:
            return ""
        
        parts = []
        for entity_type, items in self.entities.items():
            if items:
                ids = list(items.keys())[-5:]  # 最近5个
                if entity_type == "order_id":
                    parts.append(f"已提及的订单号: {', '.join(ids)}")
                elif entity_type == "refund_id":
                    parts.append(f"已提及的退款单号: {', '.join(ids)}")
        
        return "; ".join(parts) if parts else ""
    
    def add_summary(self, summary: str, turn_count: int):
        """添加会话摘要"""
        self.summaries.append({
            "timestamp": datetime.now().isoformat(),
            "turn_count": turn_count,
            "summary": summary
        })
        # 只保留最近 10 个摘要
        if len(self.summaries) > 10:
            self.summaries = self.summaries[-10:]
    
    def get_recent_summaries(self, n: int = 3) -> str:
        """获取最近 N 个会话摘要"""
        if not self.summaries:
            return ""
        
        recent = self.summaries[-n:]
        summary_texts = [s["summary"] for s in recent]
        return "\n".join(summary_texts)
    
    def save(self):
        """持久化保存长期记忆"""
        data = {
            "summaries": self.summaries,
            "entities": self.entities,
            "last_saved": datetime.now().isoformat()
        }
        
        filepath = self.storage_path / "long_term_memory.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 长期记忆已保存到 {filepath}")
    
    def load(self):
        """加载长期记忆"""
        filepath = self.storage_path / "long_term_memory.json"
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.summaries = data.get("summaries", [])
                self.entities = data.get("entities", {})
                print(f"📂 已加载长期记忆 (摘要: {len(self.summaries)}, 实体: {sum(len(v) for v in self.entities.values())})")
            except Exception as e:
                print(f"⚠️ 加载长期记忆失败: {e}")


class MemoryManager:
    """
    统一记忆管理器
    
    整合短期记忆和长期记忆，提供统一的接口给 Agent 使用
    """
    
    def __init__(self, max_short_term_turns: int = 10, storage_path: str = "./memory_store"):
        """
        Args:
            max_short_term_turns: 短期记忆保留的最大对话轮数
            storage_path: 长期记忆存储路径
        """
        self.short_term = ConversationMemory(max_turns=max_short_term_turns)
        self.long_term = SummaryMemory(storage_path=storage_path)
        self._llm = None  # 用于生成摘要的 LLM (延迟设置)
    
    def set_llm(self, llm):
        """设置用于生成摘要的 LLM"""
        self._llm = llm
    
    def load(self):
        """加载持久化的长期记忆"""
        self.long_term.load()
    
    def save(self):
        """保存长期记忆"""
        self.long_term.save()
    
    def add_turn(self, user_message: str, assistant_message: str):
        """添加一轮对话到短期记忆，并提取实体"""
        self.short_term.add(user_message, assistant_message)
        
        # 从对话中提取实体到长期记忆
        self.long_term.remember_entities(user_message)
        self.long_term.remember_entities(assistant_message)
    
    def get_full_context(self) -> list[dict]:
        """
        获取完整的记忆上下文
        
        结构:
        1. 长期记忆摘要 (如果有)
        2. 已知实体 (如果有)
        3. 短期对话历史
        """
        context = []
        
        # 1. 添加长期记忆摘要
        summaries = self.long_term.get_recent_summaries(n=2)
        entities = self.long_term.get_known_entities()
        
        if summaries or entities:
            memory_hint = []
            if summaries:
                memory_hint.append(f"[历史对话摘要]\n{summaries}")
            if entities:
                memory_hint.append(f"[已知实体] {entities}")
            
            context.append({
                "role": "system",
                "content": "\n".join(memory_hint)
            })
        
        # 2. 添加短期对话历史
        context.extend(self.short_term.get_context())
        
        return context
    
    def end_session(self, generate_summary: bool = True):
        """
        结束当前会话
        
        - 生成会话摘要 (如果设置了 LLM)
        - 保存长期记忆
        - 清空短期记忆
        """
        turn_count = len(self.short_term)
        
        if turn_count == 0:
            return
        
        # 生成简单摘要 (不使用 LLM 的简化版)
        if generate_summary and turn_count >= 2:
            # 简单摘要: 提取用户问过的问题
            user_questions = [
                m["content"][:50] + "..." if len(m["content"]) > 50 else m["content"]
                for m in self.short_term.history 
                if m["role"] == "user"
            ]
            summary = f"用户咨询了 {turn_count} 个问题，包括: " + "; ".join(user_questions[-3:])
            self.long_term.add_summary(summary, turn_count)
        
        # 保存并清空
        self.save()
        self.short_term.clear()
        print(f"🧹 会话结束，短期记忆已清空")
    
    def get_memory_info(self) -> str:
        """获取记忆系统状态信息"""
        short_len = len(self.short_term)
        summary_count = len(self.long_term.summaries)
        entity_count = sum(len(v) for v in self.long_term.entities.values())
        
        return f"📊 记忆状态: 短期={short_len}轮, 长期摘要={summary_count}, 实体={entity_count}"
