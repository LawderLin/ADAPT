from config import config
import ollama
import json
import re
from typing import List, Dict, Any

class ItemWritingAgent():
    """
    基於 Ollama 的題目生成代理人，根據使用者規格使用指定語言模型生成測驗題目。
    """

    def __init__(self, model: str = None, system_prompt: str = None):
        self.model = model or config.model_name
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """預設的系統提示詞"""
        return """你是一個專業的心理測驗題目編寫專家。請根據使用者的需求，生成高品質的測驗題目。

生成規則：
1. 每個題目都應該有明確的心理學理論基礎
2. 題目語言應該清晰、無歧義
3. 避免文化偏見和人口學偏見
4. 題目難度應該適中
5. 提供多個選項

輸出格式：
請以 JSON 格式輸出，包含以下欄位：
- item: 題目內容
- psychological_construct: 心理建構
"""

    def generate_items(self, specifications: str, num_items: int = 5) -> Dict[str, Any]:
        """
        根據規格生成測驗題目

        Args:
            specifications (str): 使用者對測驗題目的規格要求
            num_items (int): 要生成的題目數量

        Returns:
            Dict[str, Any]: 生成的題目資料，包含原始訊息與處理後的JSON題目列表
        """
        try:
            # 構建完整的提示詞
            user_prompt = f"""
規格要求：
{specifications}

請根據以上規格生成 {num_items} 個心理測驗題目，嚴格按照 JSON 格式輸出結果。
"""

            # 調用 Ollama API
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens
                },
                # format="json"
            )

            # 解析回應
            content = response['message']['content']

            result = self.process_json_response(content)

            return result

        except Exception as e:
            return {
                "error": f"生成題目時發生錯誤: {str(e)}",
                "items": []
            }

    def refine_items(self, items: List[str], feedback: str) -> Dict[str, Any]:
        """
        根據回饋改進題目

        Args:
            items: 原始題目列表
            feedback: 改進建議

        Returns:
            改進後的題目
        """
        try:
            refine_prompt = f"""
請根據以下回饋改進這些心理測驗題目：

原始要求：

原始題目：
{json.dumps(items, ensure_ascii=False, indent=2)}

改進建議：
{feedback}

請輸出改進後的題目（JSON 格式）。
"""

            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": refine_prompt}
                ],
                options={
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens
                }
            )

            content = response['message']['content']

            result = self.process_json_response(content)

            return result

        except Exception as e:
            return {
                "error": f"改進題目時發生錯誤: {str(e)}",
                "items": items
            }

    def process_json_response(self, content: str) -> Dict[str, Any]:
        """
        處理模型回應的 JSON 格式

        Args:
            content: 模型回應內容

        Returns:
            解析後的 JSON 資料
        """
        try:
            # 先去除 markdown 格式
            content = re.sub(r"^```json|^```|```$", "",
                             content, flags=re.MULTILINE).strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL).group()
            if json_match:
                content = json.loads(f"[{json_match}]")
                # content = json.loads(json_match)
                result = {"raw_output": content, "items": content}
            else:
                result = {"raw_output": content, "items": content}
        except json.JSONDecodeError:
            result = {"raw_output": content, "items": content}

        return result


# 建立題目生成器實例
item_writer = ItemWritingAgent()
print("✅ ItemWritingAgent 已初始化完成！")