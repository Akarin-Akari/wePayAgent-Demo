#!/usr/bin/env python3
"""
RAG 评测脚本
=================================
使用 LLM-as-a-Judge 模式对 RAG 系统进行自动化评测。

评测维度：
1. 关键词覆盖率 (Keyword Hit Rate) - 硬指标
2. LLM 语义打分 (Semantic Score) - 软指标 (0-10分)
"""

import json
import sys
import os
import argparse
from pathlib import Path
from rag_demo import (
    load_documents, 
    chunk_documents, 
    OllamaEmbedding, 
    OllamaLLM, 
    SimpleVectorStore, 
    RAGPipeline
)

def evaluate_answer(judge_llm, question, expected, actual) -> dict:
    """使用 LLM 对问答质量进行打分"""
    prompt = f"""作为公平的评测员，请评估以下AI回答的质量。

【用户问题】{question}
【标准答案】{expected}
【AI 回答】{actual}

请根据AI回答是否准确包含了标准答案的核心信息进行打分（0-10分）。
0分：完全错误或未回答
5分：部分正确，但有遗漏
10分：完全正确且表述清晰

请仅返回 JSON 格式结果，格式如下：
{{
    "score": 8,
    "reason": "回答准确，覆盖了核心点"
}}
"""
    try:
        # 强制 LLM 输出 JSON
        response_str = judge_llm.generate(prompt + "\n\n请只输出JSON格式 (例如: {\"score\": 8, \"reason\": \"...\"})，不要包含Markdown或其他文字。")
        
        # 尝试清理 markdown
        clean_text = response_str.replace("```json", "").replace("```", "").strip()
        
        # 尝试直接解析
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            pass

        # 如果 JSON 解析失败，尝试从文本中提取分数
        import re
        score_match = re.search(r'"score":\s*(\d+)', clean_text)
        if not score_match:
             score_match = re.search(r'score:\s*(\d+)', clean_text)
        
        score = int(score_match.group(1)) if score_match else 0
        
        # 提取 Reason (简单处理)
        reason = "Parsed from text"
        if "reason" in clean_text:
            parts = clean_text.split("reason")
            if len(parts) > 1:
                reason = parts[1].strip().strip('":,').strip()
        
        return {"score": score, "reason": reason}

    except Exception as e:
        print(f"⚠️ 评测打分失败: {e} | Raw: {response_str[:50]}...")
        return {"score": 5, "reason": "Evaluator Parsing Failed (Default 5)"}

def main():
    # 1. 初始化 Pipeline
    print("🚀 初始化 RAG Pipeline...")
    knowledge_dir = "./knowledge_base"
    docs = load_documents(knowledge_dir)
    chunks = chunk_documents(docs)
    
    # 强制做一次离线索引
    embed_model = OllamaEmbedding(model="bge-m3")
    llm = OllamaLLM(model="qwen3:4b")
    vector_store = SimpleVectorStore()
    
    pipeline = RAGPipeline(embed_model, llm, vector_store)
    pipeline.index_documents(chunks)
    
    # 2. 加载测试集
    data_path = Path("./data/benchmark_qa.json")
    if not data_path.exists():
        print(f"❌ 未找到测试集: {data_path}")
        return
        
    with open(data_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    print(f"\n🧪 开始评测 (共 {len(test_cases)} 个测试用例)...\n")
    
    total_score = 0
    total_keyword_hit = 0
    total_expected_keywords = 0
    
    results = []
    
    for i, case in enumerate(test_cases):
        q = case["question"]
        expected = case["expected_answer"]
        keywords = case.get("keywords", [])
        
        print(f"[{i+1}/{len(test_cases)}] 提问: {q}")
        
        # 运行 RAG
        actual = pipeline.query(q, top_k=3)
        print(f"   🤖 回答: {actual.strip()[:60]}...")
        
        # 1. 关键词评测
        hits = sum(1 for k in keywords if k in actual)
        keyword_rate = hits / len(keywords) if keywords else 1.0
        total_keyword_hit += hits
        total_expected_keywords += len(keywords)
        
        # 2. LLM 打分
        eval_result = evaluate_answer(llm, q, expected, actual)
        score = eval_result.get("score", 0)
        total_score += score
        
        print(f"   📊 评测: 得分={score}/10 | 关键词命中={hits}/{len(keywords)}")
        print(f"   💡 理由: {eval_result.get('reason')}\n")
        
        results.append({
            "question": q,
            "actual": actual,
            "score": score,
            "keyword_rate": keyword_rate
        })

    # 3. 汇总报告
    avg_score = total_score / len(test_cases)
    avg_keyword_rate = (total_keyword_hit / total_expected_keywords * 100) if total_expected_keywords else 0
    
    print("="*40)
    print("       RAG 评测报告 Result")
    print("="*40)
    print(f"Tests: {len(test_cases)}")
    print(f"Avg Semantic Score: {avg_score:.1f} / 10")
    print(f"Keyword Coverage  : {avg_keyword_rate:.1f}%")
    print("="*40)

if __name__ == "__main__":
    main()
