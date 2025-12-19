#!/usr/bin/env python3
"""
Agent Demo Launcher
===================
启动微信支付智能客服 Agent
包含环境健壮性检查
"""

import sys
import argparse
import subprocess
import requests
from tools import get_all_tools
from agent_core import OllamaLLM, ReActAgent
from memory import MemoryManager

def check_ollama_installed() -> bool:
    """检查 Ollama 是否安装"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except Exception:
        return False

def check_ollama_running() -> bool:
    """检查 Ollama 服务是否运行"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_installed_models() -> list:
    """获取已安装的模型列表"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m.get("name", "") for m in data.get("models", [])]
    except:
        pass
    return []

def pull_model(model_name: str) -> bool:
    """下载指定模型"""
    print(f"📥 正在下载模型 {model_name}...")
    print("   (首次下载可能需要几分钟，请耐心等待)")
    try:
        # 使用 subprocess 运行 ollama pull
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=False,  # 显示下载进度
            timeout=600  # 10分钟超时
        )
        return result.returncode == 0
    except Exception as e:
        print(f"⚠️ 下载失败: {e}")
        return False

def environment_check(model_name: str) -> bool:
    """环境健壮性检查"""
    print("🔍 正在检查运行环境...")
    
    # 1. 检查 Ollama 是否安装
    if not check_ollama_installed():
        print("❌ Ollama 未安装！")
        print("   请访问 https://ollama.ai 下载安装 Ollama")
        print("   Windows: winget install Ollama.Ollama")
        print("   或下载安装包: https://ollama.ai/download/windows")
        return False
    print("✅ Ollama 已安装")
    
    # 2. 检查 Ollama 服务是否运行
    if not check_ollama_running():
        print("⚠️ Ollama 服务未运行，正在尝试启动...")
        try:
            # Windows 下尝试启动 Ollama
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            import time
            time.sleep(3)  # 等待服务启动
            if not check_ollama_running():
                print("❌ 无法启动 Ollama 服务，请手动运行: ollama serve")
                return False
        except Exception as e:
            print(f"❌ 启动 Ollama 服务失败: {e}")
            print("   请手动运行: ollama serve")
            return False
    print("✅ Ollama 服务运行中")
    
    # 3. 检查模型是否已下载
    installed_models = get_installed_models()
    model_base = model_name.split(":")[0]  # qwen3:4b -> qwen3
    
    model_found = any(model_name in m or model_base in m for m in installed_models)
    
    if not model_found:
        print(f"⚠️ 模型 {model_name} 未找到")
        print(f"   已安装的模型: {installed_models if installed_models else '无'}")
        
        # 询问是否下载
        try:
            choice = input(f"   是否现在下载 {model_name}? (y/n): ").strip().lower()
            if choice == 'y':
                if pull_model(model_name):
                    print(f"✅ 模型 {model_name} 下载完成！")
                else:
                    print(f"❌ 模型下载失败，请手动运行: ollama pull {model_name}")
                    return False
            else:
                print("   请先手动下载模型后再运行")
                return False
        except KeyboardInterrupt:
            print("\n   取消下载")
            return False
    else:
        print(f"✅ 模型 {model_name} 已就绪")
    
    print("✅ 环境检查通过！\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="微信支付智能客服 Agent")
    parser.add_argument("--model", type=str, default="qwen3:4b", help="Ollama 模型名称")
    parser.add_argument("--skip-check", action="store_true", help="跳过环境检查")
    args = parser.parse_args()

    # 环境检查
    if not args.skip_check:
        if not environment_check(args.model):
            sys.exit(1)

    print(f"🚀 正在初始化智能客服 Agent (Model: {args.model})...")
    
    # 1. 初始化工具
    print("🔧 加载工具箱...")
    tools = get_all_tools()
    
    # 2. 初始化 LLM
    llm = OllamaLLM(model=args.model)
    
    # 3. 初始化记忆系统
    print("🧠 加载记忆系统...")
    memory = MemoryManager(storage_path="./memory_store")
    memory.load()  # 加载持久化的长期记忆
    
    # 4. 初始化 Agent (带记忆)
    agent = ReActAgent(llm, tools, memory=memory)
    
    print("\n✅ Agent 就绪! (输入 quit 退出)")
    print("💡 您是不是想问 '微信退款一般多久到账？' 或 '查询微信支付订单 ORDER_1001'？")
    print("🧠 记忆系统已启用，您可以说'刚才那个订单'来引用之前的对话！")
    print(f"{memory.get_memory_info()}")
    print("-" * 50)
    
    # 4. 交互循环
    while True:
        try:
            user_input = input("\n👤我: ").strip()
            if not user_input: continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 正在保存记忆...")
                memory.end_session()  # 保存记忆
                print("👋 再见！")
                break
            
            answer = agent.chat(user_input)
            print(f"\n🤖 智能客服: {answer}")
            
        except KeyboardInterrupt:
            print("\n👋 正在保存记忆...")
            memory.end_session()  # 保存记忆
            print("👋 再见！")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()

