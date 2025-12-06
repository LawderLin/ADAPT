import ollama
import json
import re
from typing import List, Dict, Any


class ContentReviewer:
    """內容效度評審員"""

    def __init__(self, model: str):
        self.model = model

    def review(self, items: List[str], construct: str) -> Dict[str, Any]:
        """評估內容效度"""
        prompt = f"""
    You are an expert in psychological test content validity. Please evaluate whether the following items effectively measure the specified psychological construct.

    Psychological construct: {construct}
    Item list:
    {json.dumps(items, ensure_ascii=False, indent=2)}

    Please assess:
    1. Whether the items are relevant to the psychological construct
    2. Whether the items cover important aspects of the construct
    3. Whether the items have sufficient theoretical foundation

    Please output your evaluation in JSON format, including:
    - validity_score: Validity score (1-10)
    - strengths: List of strengths
    - weaknesses: List of weaknesses
    - suggestions: Suggestions for improvement
    """

        return self._get_review_response(prompt)

    def _get_review_response(self, prompt: str) -> Dict[str, Any]:
        """獲取評審回應"""
        print("📝 內容效度評審中...")
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
                # format="json"
            )

            content = response['message']['content']

            # 嘗試解析 JSON
            try:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = {"raw_output": content, "validity_score": 5}
            except json.JSONDecodeError:
                result = {"raw_output": content, "validity_score": 5}

            return result

        except Exception as e:
            return {"error": str(e), "validity_score": 0}
