# RAG实战Demo项目

这是一个模拟微信支付智能客服的RAG演示项目。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置API Key（选择一个即可）
# 方式A：使用OpenAI
set OPENAI_API_KEY=sk-xxx

# 方式B：使用免费的本地模型（无需API Key）
# 安装ollama后拉取模型即可

# 3. 运行Demo
python rag_demo.py
```

## 项目结构

```
rag_demo/
├── README.md           # 本文件
├── requirements.txt    # Python依赖
├── knowledge_base/     # 知识库文档
│   └── wxpay_faq.txt   # 微信支付FAQ
├── rag_demo.py         # 主程序
└── rag_native.py       # 原生实现（无框架）
```

## 面试加分话术

> "我曾经自己动手实现过一个RAG Demo，用的是LangChain + Chroma，模拟了微信支付客服场景。整个流程包括文档分块、Embedding、向量检索、Prompt拼接和LLM生成。"
