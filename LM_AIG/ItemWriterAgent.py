from config import config
import ollama
import json
import re
from typing import List, Dict, Any
from LM_AIG.JSONFormatAgent import JSONFormatAgent

class ItemWritingAgent():
    """
    基於 Ollama 的題目生成代理人，根據使用者規格使用指定語言模型生成測驗題目。
    """

    def __init__(self, model: str = None, system_prompt: str = None,):
        self.model = model or config.model_name
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.language = config.language
        self.json_agent = JSONFormatAgent()

    def _default_system_prompt(self) -> str:
        """預設的系統提示詞"""
        return """You are a professional expert in writing psychological test items. Please generate high-quality test items according to the user's requirements.

        Generation rules:
        1. Each item should have a clear psychological theory basis.
        2. The language of the items should be clear and unambiguous.
        3. Avoid cultural and demographic biases.
        4. The difficulty of the items should be moderate.
        5. Provide multiple options for each item.

        Output format:
        Please output in JSON format, including the following fields:
        - item: The content of the test item
        - psychological_construct: The psychological construct
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
            Specifications for the psychological test items:
            {specifications}

            Please generate {num_items} psychological test items based on the above specifications, strictly outputting the results in JSON format.

            The items should be written in {self.language}
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

            result = self.json_agent.format_to_json(content)

            return result

        except Exception as e:
            return {
                "error": f"生成題目時發生錯誤: {str(e)}",
                "items": []
            }

    def refine_items(self, items: List[str], feedback: str, specifications: str, num_items: int = 5) -> Dict[str, Any]:
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
Refine the following psychological test items based on the feedback provided.

Specifications for the psychological test items:
{specifications}

Number of items: Please generate at least {num_items} items.

Original Items:
{json.dumps(items, ensure_ascii=False, indent=2)}

Feedback:
{feedback}

Please output the refined items in JSON format.
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
            return self.json_agent.format_to_json(content)

        except Exception as e:
            return {
                "error": f"改進題目時發生錯誤: {str(e)}",
                "items": items
            }


# 建立題目生成器實例
item_writer = ItemWritingAgent()
print("✅ ItemWritingAgent 已初始化完成！")