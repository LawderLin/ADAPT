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
你是語言學專家。請評估以下測驗題目的語言品質。

題目列表:
{json.dumps(items, ensure_ascii=False, indent=2)}

請評估：
1. 語言是否清晰易懂
2. 是否有語法錯誤
3. 用詞是否恰當
4. 是否有歧義表達

請以 JSON 格式輸出評估結果，包含：
- readability_score: 可讀性分數 (1-10)
- grammar_issues: 語法問題列表
- clarity_issues: 清晰度問題列表
- suggestions: 語言改進建議
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
