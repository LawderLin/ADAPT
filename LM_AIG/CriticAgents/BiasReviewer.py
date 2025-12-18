import ollama
from typing import List, Dict, Any
from pydantic import BaseModel
from LM_AIG.JSONFormatAgent import JSONFormatAgent


class BiasReviewer:
    """偏見檢查評審員"""

    def __init__(self, model: str):
        self.model = model
        self.schema = self._define_schema()

    def _define_schema(self) -> BaseModel:
        """定義輸出 JSON 結構"""
        class ReviewSchema(BaseModel):
            bias_score: int
            detected_biases: List[str]
            problematic_items: List[str]
            suggestions: List[str]

        return ReviewSchema

    def review(self, items: List[dict]) -> Dict[str, Any]:
        """檢查人口學偏見"""

        prompt = f"""You are an expert in psychological test development. Please review the following test items for potential demographic biases.

Items:
{'\n'.join([f"- {item['item']}" for item in items])}

Please check for the following biases:
1. Gender bias
2. Age bias
3. Cultural bias
4. Social economic status bias
5. Other discriminatory content

Please provide the review results, including:
- bias_score: Degree of bias (1-10, with 10 indicating no bias)
- detected_biases: List of detected bias types
- problematic_items: List of items with potential biases
- suggestions: Suggestions for eliminating biases

Provide the response in the following JSON format:

"""
        prompt += """{
  "bias_score": int,
  "detected_biases": [str],
  "problematic_items": [str],
  "suggestions": list[str]
}"""
        return self._get_review_response(prompt)

    def _get_review_response(self, prompt: str) -> Dict[str, Any]:
        """獲取評審回應"""
        print("📝 偏見檢查評審中...")
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
            return {"error": str(e), "bias_score": 0}
