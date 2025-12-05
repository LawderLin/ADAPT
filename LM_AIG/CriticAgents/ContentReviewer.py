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
你是心理測驗內容效度專家。請評估以下題目是否能有效測量指定的心理建構。

心理建構: {construct}
題目列表:
{json.dumps(items, ensure_ascii=False, indent=2)}

請評估：
1. 題目是否與心理建構相關
2. 題目是否涵蓋該建構的重要面向
3. 題目的理論基礎是否充分

請以 JSON 格式輸出評估結果，包含：
- validity_score: 效度分數 (1-10)
- strengths: 優點列表
- weaknesses: 缺點列表
- suggestions: 改進建議
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
