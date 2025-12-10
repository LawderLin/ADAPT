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

    def format_to_json(self, data: str) -> List[Dict[str, Any]] :
        """
        檢查字串資料內容並轉換為 JSON 格式

        Args:
            data: 要格式化的資料字串

        Returns:
            result: 包含轉換後題目的 JSON 資料
        """
        
        try:
            # 先去除 markdown 格式 (如有)
            removed_markdown_data = re.sub(r"^```json|^```|```$", "",
                             data, flags=re.MULTILINE).strip()
            json_match = re.search(r'\{.*\}', removed_markdown_data, re.DOTALL)
            if json_match is not None:
                json_match = json_match.group(0)
                result = json.loads(f"[{json_match}]")
            else:
                # 如果原始資料為空，直接回傳錯誤訊息
                if data.strip() == "":
                    raise json.JSONDecodeError("Empty data", data, 0)
                # 如果無法找到 JSON，則使用 LLM 進行改進
                result = self.LLM_refine_to_json(data)
        except json.JSONDecodeError:
            if data.strip() == "":
                return {
                    "error": "資料為空，無法轉換為 JSON 格式。",
                    "items": []
                }
            print("❌ JSON 解碼錯誤，嘗試使用 LLM 進行改進...")
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
        If the original data is already in valid JSON format, return it directly.
        Do not fabricate any information; only use the data provided.
        **Only** when the original data does not contain any valid JSON, extract useful information and convert it into JSON format.
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
            result = json.loads(content)

        except Exception as e:
            return {
                "error": f"修正資料 JSON 格式時發生錯誤: {str(e)}",
                "items": items
            }
        return result