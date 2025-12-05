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
你是測驗評審的資深專家。請整合以下各個評審員的評審結果，給出綜合評價和改進建議。

原始題目:
{json.dumps(items, ensure_ascii=False, indent=2)}

各評審員結果:
內容效度評審: {json.dumps(reviews.get('content_review', {}), ensure_ascii=False, indent=2)}
語言學評審: {json.dumps(reviews.get('linguistic_review', {}), ensure_ascii=False, indent=2)}
偏見檢查評審: {json.dumps(reviews.get('bias_review', {}), ensure_ascii=False, indent=2)}

請提供：
1. 綜合評分 (1-10)
2. 主要優點
3. 主要問題
4. 優先改進建議
5. 是否建議重新生成

請以 JSON 格式輸出，包含：
- overall_score: 綜合分數
- strengths: 主要優點列表
- major_issues: 主要問題列表
- recommendations: 改進建議列表
- regenerate_recommended: 是否建議重新生成 (boolean)
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
