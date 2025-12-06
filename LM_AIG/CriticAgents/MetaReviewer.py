import ollama
import json
import re
from typing import List, Dict, Any


class MetaReviewer:
    """元評審員，整合所有評審結果"""

    def __init__(self, model: str):
        self.model = model

    def integrate_reviews(self, reviews: Dict[str, Any], items: List[str]) -> Dict[str, Any]:
        """整合所有評審結果"""
        prompt = f"""
    You are a senior expert in test item review. Please integrate the following review results from different reviewers and provide an overall evaluation and suggestions for improvement.

    Original items:
    {json.dumps(items, ensure_ascii=False, indent=2)}

    Review results from each reviewer:
    Content validity review: {json.dumps(reviews.get('content_review', {}), ensure_ascii=False, indent=2)}
    Linguistic review: {json.dumps(reviews.get('linguistic_review', {}), ensure_ascii=False, indent=2)}
    Bias check review: {json.dumps(reviews.get('bias_review', {}), ensure_ascii=False, indent=2)}

    Please provide:
    1. Overall score (1-10)
    2. Main strengths
    3. Major issues
    4. Priority recommendations for improvement
    5. Whether you recommend regeneration

    Please output in JSON format, including:
    - overall_score: Overall score
    - strengths: List of main strengths
    - major_issues: List of major issues
    - recommendations: List of recommendations for improvement
    - regenerate_recommended: Whether regeneration is recommended (boolean)
    """
        return self._get_review_response(prompt)

    def _get_review_response(self, prompt: str) -> Dict[str, Any]:
        """獲取評審回應"""
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
                    result = {"raw_output": content, "overall_score": 5}
            except json.JSONDecodeError:
                result = {"raw_output": content, "overall_score": 5}

            return result

        except Exception as e:
            return {"error": str(e), "overall_score": 0}
