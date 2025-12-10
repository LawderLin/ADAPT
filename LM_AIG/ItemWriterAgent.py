from config import config
import ollama
import json
from pydantic import BaseModel
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
        self.schema = self._define_schema()
        self.json_agent = JSONFormatAgent()

    def _default_system_prompt(self) -> str:
        """預設的系統提示詞"""
        return """You are a professional expert in writing psychological test items. Please generate high-quality test items according to the user's requirements.

        Generation rules:
        1. Each item should have a clear psychological theory basis.
        2. The language of the items should be clear and unambiguous.
        3. Avoid cultural and demographic biases.
        4. The difficulty of the items should be moderate.
        5. The items should be Likert-scale questions.

        Output format:
        Please output in JSON format, including the following fields:
        - item: The content of the test item
        - psychological_construct: The psychological construct
        """
    
    def _define_schema(self) -> BaseModel:
        """定義輸出 JSON 結構"""
        class ItemSchema(BaseModel):
            item: str
            psychological_construct: str

        return ItemSchema

    def generate_items(self, specifications: str, num_items: int = 5, construct_definitions: str = "") -> List[Dict[str, Any]]:
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

            - The items should be written in {self.language}
            - Please generate at least {num_items} items.
            """

            if construct_definitions != "":
                user_prompt += f"\n\nPsychological construct definitions:\n{construct_definitions}\n"

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
                format=self.schema.model_json_schema()
            )

            # 解析回應
            content = response['message']['content']

            result = self.json_agent.format_to_json(content)

            try:
                item = result[0]['item']
            except KeyError:
                print("❌ 無法從回應中提取題目，正在重新嘗試生成題目...")
                return self.generate_items(specifications, num_items, construct_definitions)

            return result

        except Exception as e:
            # 若是無法找到 JSON 的錯誤，直接重新嘗試生成
            retry = [
                "\'NoneType\' object has no attribute \'group\'",
                "修正資料 JSON 格式時發生錯誤: Expecting value: line 1 column 1 (char 0)"
            ]
            if str(e) in retry:
                print("❌ 未能生成有效的 JSON，正在重新嘗試生成題目...")
                return self.generate_items(specifications, num_items, construct_definitions)
            else:
                print(f"❌ 生成題目時發生錯誤: {str(e)}")
                exit(1)
                # return {
                #     "error": f"生成題目時發生錯誤: {str(e)}",
                #     "items": []
                # }

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

Original Items:
{'\n'.join([f"- {item['item']} ({item['psychological_construct']})" for item in items])}

Feedback:
{'\n'.join([f"- {f}" for f in feedback])}

- The items should be written in {self.language}
- Please generate at least {num_items} items. (You may modify existing items or create new ones based on the feedback.)

Please output the refined items list in JSON format.
"""
            print("提示詞：", refine_prompt)
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": refine_prompt}
                ],
                options={
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens
                },
                format=self.schema.model_json_schema()
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