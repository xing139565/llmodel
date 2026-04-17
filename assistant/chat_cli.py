import requests
import json
import sys

def main():
    print("=== 欢迎测试本地 RAG 知识库系统 ===")
    print("请输入你的问题（输入 q 或 exit 退出）：\n")
    
    url = "http://localhost:8000/chat"
    
    history = []
    
    while True:
        question = input("问题 > ")
        if question.lower() in ['q', 'exit', 'quit']:
            break
            
        if not question.strip():
            continue
            
        print("\n【回答】: ", end='', flush=True)
        try:
            sources = []
            full_answer = ""
            
            with requests.post(
                url, 
                json={
                    "question": question,
                    "history": history
                },
                stream=True,
                timeout=120
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        if data.get("type") == "meta":
                            sources = data.get("sources", [])
                            route_type = data.get("route_type", "未知")
                            # 隐式提示，不打扰正文
                        elif data.get("type") == "chunk":
                            chunk_text = data.get("content", "")
                            print(chunk_text, end='', flush=True)
                            full_answer += chunk_text
                            
            print("\n")
            if sources:
                print(f"【参考文档来源】:\n{', '.join(sources)}")
            print("-" * 50)
            
            # 本地维护对话历史
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": full_answer})
            if len(history) > 6:  # 保持最新的3轮 (6条消息)
                history = history[-6:]

            
        except requests.exceptions.ConnectionError:
            print("连接失败！请确认你已经运行了 python -m uvicorn app:app --host 0.0.0.0 --port 8000启动服务。")
        except Exception as e:
            print(f"\n发生错误: {e}")

if __name__ == "__main__":
    main()
