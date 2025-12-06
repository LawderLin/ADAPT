# 導入必要的套件
import ollama
from dataclasses import dataclass

# 設定基本配置
@dataclass
class Config:
    """系統配置類別"""
    model_name: str = "qwen3:8b"  # 使用較小的模型
    temperature: float = 0.7
    max_tokens: int = 1000
    language: str = "繁體中文"

config = Config()

print("🚀 AI 自動出題系統初始化完成！")
print(f"使用模型: {config.model_name}")

# 檢查 Ollama 服務是否運行
try:
    models_response = ollama.list()
    print(f"✅ Ollama 服務正常")

    # 檢查模型是否存在
    available_models = []
    if 'models' in models_response:
        available_models = [model.get('name', '')
                            for model in models_response['models']]

    print(f"已安裝模型: {available_models}")

    # 測試模型調用
    test_response = ollama.chat(
        model=config.model_name,
        messages=[
            {"role": "user", "content": "Hello, please respond with 'System ready'"}],
        options={"temperature": 0.1, "num_predict": 10}
    )

    if test_response and 'message' in test_response:
        print(f"✅ 模型 {config.model_name} 測試成功！")
        print(f"測試回應: {test_response['message']['content']}")
    else:
        print("⚠️  模型回應格式異常")

except Exception as e:
    print(f"❌ Ollama 連接失敗: {e}")
    print("請確保 Ollama 服務正在運行並且模型已安裝")
    print("可以嘗試運行: ollama pull llama3.2:1b")