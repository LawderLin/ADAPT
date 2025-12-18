# 導入必要的套件
import ollama
from dataclasses import dataclass

# 設定基本配置


@dataclass
class Config:
    """系統配置類別"""
    model_name: str = "qwen3:8b"  # 使用較小的模型
    temperature: float = 0.7
    max_tokens: int = 1000 # 增加最大生成長度
    language: str = "繁體中文" # 字串會傳進 prompt 中，所以直接敘述要使用的語言就好

    def select_model(self, model_name: str):
        """選擇使用的模型"""
        self.model_name = model_name


config = Config()

print("環境初始化完成！")

# 檢查 Ollama 服務是否運行
models_response = {}
try:
    models_response = ollama.Client().list()
    print(f"✅ Ollama 服務正常")
except Exception as e:
    print(f"❌ Ollama 連接失敗: {e}")
    print("請確保 Ollama 服務正在運行")
    print(f"可以嘗試運行:  ollama serve")
    exit(1)

try:
    # 檢查模型是否存在
    available_models = []
    if 'models' in models_response:
        available_models = [model['model']
                            for model in models_response['models']]
    else:
        available_models = []

    print(f"已安裝模型: {available_models}")
    print(f"使用模型: {config.model_name}")
    if config.model_name in available_models:
        print(f"✅ 模型 {config.model_name} 已安裝")
    else:
        print(f"❌ 模型 {config.model_name} 未安裝，請先安裝該模型或設定使用其他模型")
        print(f"可以嘗試運行: ollama pull {config.model_name}")
        exit(1)

    if config.model_name in available_models:
        print(f"✅ 模型 {config.model_name} 已安裝")
    else:
        print(f"❌ 模型 {config.model_name} 未安裝，請先安裝該模型或設定使用其他模型")
        print(f"可以嘗試運行: ollama pull {config.model_name}")
        exit(1)

    # 測試模型調用
    print(f"正在測試模型 {config.model_name} 的回應...")
    test_response = ollama.chat(
        model=config.model_name,
        messages=[
            {"role": "user", "content": "Hello, please respond with 'System ready!'"}],
        options={"temperature": 0.1, "num_predict": 10},
        think=False, # 測試時不啟用思考模式，以加快回應速度
    )

    if test_response and 'message' in test_response:
        print(f"✅ 模型 {config.model_name} 測試成功！")
    else:
        print("⚠️  模型回應格式異常")
        print("請確認模型是否正確安裝並可用，或 ollama 版本是否為最新")
        exit(1)

except Exception as e:
    print(f"❌ Ollama 連接異常: {e}")
    print("請確保 Ollama 服務正在運行並且模型已安裝")
    print(f"可以嘗試運行: ollama pull {config.model_name}")
    exit(1)
