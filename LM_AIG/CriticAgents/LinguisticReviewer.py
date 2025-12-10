import ollama
from pydantic import BaseModel
from typing import List, Dict, Any

from LM_AIG.JSONFormatAgent import JSONFormatAgent

class LinguisticReviewer:
    """語言學評審員"""

    def __init__(self, model: str):
        self.model = model
        self.schema = self._define_schema()

    def _define_schema(self) -> BaseModel:
        """定義輸出 JSON 結構"""
        class ReviewSchema(BaseModel):
            linguistic_score: int
            grammar_issues: List[str]
            clarity_issues: List[str]
            suggestions: List[str]

        return ReviewSchema

    def review(self, items: List[dict]) -> Dict[str, Any]:
        """評估題目的語言品質"""

        prompt = f"""
    You are a linguistics expert. Please evaluate the language quality of the following test items.

    Item list:
    {'\n'.join([f"- {item['item']}" for item in items])}

    Please assess:
    1. Is the language clear and easy to understand?
    2. Are there any grammatical errors?
    3. Is the wording appropriate?
    4. Are there any ambiguous expressions?

    Please output the evaluation results, including:
    - linguistic_score: readability score (1-10)
    - grammar_issues: list of grammar issues
    - clarity_issues: list of clarity issues
    - suggestions: list of suggestions for language improvement

    If there isn't any issues, please state "No issues found". Don't force to make up issues.

    Provide the response in the following JSON format:
    """
        
        prompt += """{
            "linguistic_score": int,
            "grammar_issues": list[str],
            "clarity_issues": list[str],
            "suggestions": list[str]
    }"""

        return self._get_review_response(prompt)

    def _get_review_response(self, prompt: str) -> Dict[str, Any]:
        """獲取評審回應"""
        print("📝 語言學評審中...")
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
                format=self.schema.model_json_schema()
            )

            content = response['message']['content']
            result = JSONFormatAgent().format_to_json(content)
            return result[0]

        except Exception as e:
            print("Failed to parse reviews. Retrying...")
            return self._get_review_response(prompt)
