import ollama
import json
import re
from typing import List, Dict, Any

class LinguisticReviewer:
    """語言學評審員"""

    def __init__(self, model: str):
        self.model = model

    def review(self, items: List[str]) -> Dict[str, Any]:
        """評估題目的語言品質"""
        prompt = f"""
    You are a linguistics expert. Please evaluate the language quality of the following test items.

    Item list:
    {json.dumps(items, ensure_ascii=False, indent=2)}

    Please assess:
    1. Is the language clear and easy to understand?
    2. Are there any grammatical errors?
    3. Is the wording appropriate?
    4. Are there any ambiguous expressions?

    Please output the evaluation results in JSON format, including:
    - readability_score: readability score (1-10)
    - grammar_issues: list of grammar issues
    - clarity_issues: list of clarity issues
    - suggestions: suggestions for language improvement
    """

        return self._get_review_response(prompt)

    def _get_review_response(self, prompt: str) -> Dict[str, Any]:
        """獲取評審回應"""
        print("📝 語言學評審中...")
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
                # format="json"
            )

            content = response['message']['content']

            try:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = {"raw_output": content, "readability_score": 5}
            except json.JSONDecodeError:
                result = {"raw_output": content, "readability_score": 5}

            return result

        except Exception as e:
            return {"error": str(e), "readability_score": 0}
