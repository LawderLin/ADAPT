import ollama
from pydantic import BaseModel
from typing import List, Dict, Any

from LM_AIG.JSONFormatAgent import JSONFormatAgent

class ContentReviewer:
    """內容效度評審員"""

    def __init__(self, model: str):
        self.model = model
        self.system_prompt = self._default_system_prompt()
        self.schema = self._define_schema()

    def _define_schema(self) -> BaseModel:
        """定義輸出 JSON 結構"""
        class ReviewSchema(BaseModel):
            validity_score: int
            weaknesses: str
            suggestions: list[str]

        return ReviewSchema

    def _default_system_prompt(self) -> str:
        system_prompt = f"""
    You are an expert in psychological test content validity. Please evaluate whether the following items effectively measure the specified psychological construct.

    Please assess:
    1. Whether the items are relevant to the psychological construct
    2. Whether the items cover important aspects of the construct
    3. Whether the items have sufficient theoretical foundation
    4. Whether the items focus on the target construct without too much overlap with other constructs

    Please output your evaluation in JSON format, including:
    - validity_score: Validity score (1-10)
    - weaknesses: A string describing the weaknesses of the items
    - suggestions: Suggestions for improvement.

    If there isn't any issues, please state "No issues found". Don't force to make up issues.

    Provide the response in the following JSON format:
    """
        system_prompt += """{
            "validity_score": int,
            "weaknesses": str,
            "suggestions": list[str]
        }"""

        return system_prompt
    
    def review(self, items: List[str], construct: str) -> Dict[str, Any]:
        """評估內容效度"""

        prompt = f"""
Original specification of the psychological construct:
{construct}

Item list:
{'\n'.join([f"- {item['item']}" for item in items])}
"""
        return self._get_review_response(prompt)

    def _get_review_response(self, prompt: str) -> Dict[str, Any]:
        """獲取評審回應"""
        print("📝 內容效度評審中...")
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
                options={"temperature": 0.3},
                format=self.schema.model_json_schema()
            )

            content = response['message']['content']
            result = JSONFormatAgent().format_to_json(content)
            return result[0]

        except Exception as e:
            print("Failed to parse reviews. Retrying...")
            return self._get_review_response(prompt)
