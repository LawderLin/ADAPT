from config import config
from typing import List, Dict, Any

import ollama  
import json
import re

class JSONFormatAgent:
    """
    檢查其他 Agents 生成的資料並將其轉換為 JSON 格式
    """

    def __init__(self, model: str = None):
        self.model = model or config.model_name
        self.system_prompt = self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """預設的系統提示詞"""
        return """You are a professional data formatting agent. Your task is to check and convert the input data into JSON format.
        Generation rules:
        1. If the data is already in valid JSON format, return the JSON directly.
        2. If the data is not valid JSON, try to extract useful information and convert it into JSON format.
        3. Do not fabricate any information; only use the data provided.
        4. Ensure the output JSON is well-structured and adheres to standard JSON syntax
        """

    def format_to_json(self, data: str) -> str:
        """
        檢查字串資料內容並轉換為 JSON 格式

        Args:
            data: 要格式化的資料字串

        Returns:
            JSON 字串
        """
        
        try:
            # 先去除 markdown 格式
            data = re.sub(r"^```json|^```|```$", "",
                             data, flags=re.MULTILINE).strip()
            json_match = re.search(r'\{.*\}', data, re.DOTALL).group()
            if json_match:
                data = json.loads(f"[{json_match}]")
                # content = json.loads(json_match)
                result = {"raw_output": data, "items": data}
            else:
                # 如果無法找到 JSON，則使用 LLM 進行改進
                result = self.LLM_refine_to_json(data)
        except json.JSONDecodeError:
            result = self.LLM_refine_to_json(data)

        return result

    def LLM_refine_to_json(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        使用 LLM 將資料改進並轉換為 JSON 格式

        Args:
            items: 要改進的題目列表

        Returns:
            包含改進後題目的 JSON 資料
        """
        refine_prompt = f"""
        Format the following strings into valid JSON format.

        Original strings:
        {items}

        Please output the improved data in JSON format.
        """

        print("Using LLM to refine data into JSON format...")

        try:
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
            result = {"raw_output": content, "items": json.loads(content)}

        except Exception as e:
            return {
                "error": f"修正資料 JSON 格式時發生錯誤: {str(e)}",
                "items": items
            }
        return result