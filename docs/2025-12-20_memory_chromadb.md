# 2025-12-20 开发日志 - 记忆系统与 ChromaDB 向量存储

> 开发者：阿卡林  
> 日期：2025-12-20  
> 版本：v1.1  

---

## 📋 本次开发概述

本次开发为 wePayAgent_Demo 项目添加了两个重要功能：
1. **对话记忆系统** - 实现短期记忆和长期记忆
2. **ChromaDB 持久化向量存储** - 替代内存存储，解决大知识库问题

---

## 🧠 功能一：对话记忆系统

### 需求背景
Agent 无法记住之前的对话内容，用户说"刚才那个订单"无法关联到之前提到的订单号。

### 实现方案

#### 新增文件
- `agent_demo/memory.py` - 记忆系统核心模块

#### 核心类设计

```python
# 短期记忆：存储当前会话最近 N 轮对话
class ConversationMemory:
    def add_turn(self, user_input: str, assistant_response: str)
    def get_context(self) -> str

# 长期记忆：摘要 + 实体记忆 + JSON 持久化
class SummaryMemory:
    def add_summary(self, summary: str)
    def add_entity(self, entity_type: str, entity_id: str)
    def save() / load()

# 统一管理器
class MemoryManager:
    def get_memory_context() -> str
    def end_session()
```

#### 实体提取规则
- 订单号：`ORDER_\d+`
- 退款单号：`REF_\d+`

#### 持久化目录
```
agent_demo/
└── memory_store/
    └── long_term_memory.json
```

### 验证结果
```
💭 Thought: 用户提到"刚刚那个订单"，根据已知实体中已提及的订单号ORDER_1001，
           我需要查询该订单的最新状态。
Action: order_query
Action Input: ORDER_1001  ← 自动从记忆中关联！
```

---

## 🗄️ 功能二：ChromaDB 向量存储

### 需求背景
知识库扩充到 ~350 行后，内存存储和 prompt 都面临溢出问题。

### 实现方案

#### 新增文件
- `agent_demo/vector_store.py` - 向量存储模块
- `agent_demo/requirements.txt` - 依赖清单

#### 核心类设计

```python
# ChromaDB 兼容的 Embedding 函数
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings

# ChromaDB 持久化存储
class ChromaVectorStore:
    def add_documents(self, documents: list[str])
    def search(self, query: str, top_k: int = 3)
    def clear()
```

#### 智能分块策略
```python
def _smart_chunk(self, documents: list[str], chunk_size: int = 800):
    # 按章节标题分割，保持语义完整性
    # 新块继承章节标题作为上下文
```

#### 持久化目录
```
agent_demo/
└── chroma_db/
    └── [ChromaDB 内部文件]
```

### 关键改进
| 特性 | 说明 |
|------|------|
| 持久化 | 首次索引后，后续启动直接加载 |
| 自动去重 | 使用内容 hash 作为 ID |
| 智能分块 | 按章节分割，保持语义 |
| 相关度评分 | 返回检索结果的相似度分数 |

### 验证结果
```
📦 ChromaDB 已初始化: chroma_db
   集合: wxpay_knowledge, 现有文档: 75
✅ 正确回答"分账最高比例是多少？" → 30%
```

---

## 📚 知识库丰富

### 新增内容
从官方文档和网络搜索整合了大量信息：

| 章节 | 内容 |
|------|------|
| 退款规则 | 一年内可退、最多50次部分退款、原路退回 |
| 分账规则 | 默认最高30%、单笔最多50方、审核7个工作日 |
| 纠纷处理 | 24小时首次响应、72小时完成处理 |
| 风控解封 | 自助解冻流程、95017客服 |
| 安全机制 | 刷脸3D活体检测、指纹生物识别 |
| 商户安全 | API密钥管理、IP白名单 |

### 文件变化
- 原始：~80 行 (~3KB)
- 现在：~350 行 (~12KB)

---

## 📁 本次变更文件清单

### 新增文件
| 文件 | 说明 |
|------|------|
| `agent_demo/memory.py` | 记忆系统核心模块 |
| `agent_demo/vector_store.py` | ChromaDB 向量存储模块 |
| `agent_demo/requirements.txt` | Python 依赖清单 |
| `agent_demo/memory_store/` | 长期记忆持久化目录 |
| `agent_demo/chroma_db/` | ChromaDB 持久化目录 |
| `docs/2025-12-20_memory_chromadb.md` | 本开发文档 |

### 修改文件
| 文件 | 修改内容 |
|------|----------|
| `agent_demo/main.py` | 集成记忆系统初始化和保存 |
| `agent_demo/agent_core.py` | 集成记忆上下文到 Agent |
| `agent_demo/tools.py` | 使用 ChromaDB 替代内存存储 |
| `rag_demo/knowledge_base/wxpay_faq.txt` | 大幅丰富知识库内容 |
| `README.md` | 更新功能说明 |
| `PROJECT_OVERVIEW.md` | 更新项目结构描述 |

---

## 🔜 后续计划

- [ ] 支持更多实体类型提取（如商户号、交易时间）
- [ ] 实现基于 LLM 的智能摘要
- [ ] 添加知识库增量更新能力
- [ ] Web UI 可视化记忆状态

---

## 📝 开发备注

1. Ollama 模型路径问题：需设置 `OLLAMA_MODELS=E:\model` 环境变量
2. ChromaDB 要求 embedding 函数继承 `EmbeddingFunction` 类
3. 知识库分块建议 800 字符以内，保持语义完整
