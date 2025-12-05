import ollama
import json
import re
from typing import List, Dict, Any

class BiasReviewer:
    """偏見檢查評審員"""

    def __init__(self, model: str):
        self.model = model

    def review(self, items: List[str]) -> Dict[str, Any]:
        """檢查人口學偏見"""
        prompt = f"""
你是測驗偏見檢查專家。請檢查以下測驗題目是否存在人口學偏見。

題目列表:
{json.dumps(items, ensure_ascii=False, indent=2)}

請檢查是否存在以下偏見：
1. 性別偏見
2. 年齡偏見  
3. 文化偏見
4. 社經地位偏見
5. 其他歧視性內容

請以 JSON 格式輸出檢查結果，包含：
- bias_score: 偏見程度 (1-10, 10表示無偏見)
- detected_biases: 發現的偏見類型列表
- problematic_items: 有問題的題目
- suggestions: 消除偏見的建議
"""

        return self._get_review_response(prompt)

    def _get_review_response(self, prompt: str) -> Dict[str, Any]:
        """獲取評審回應"""
        print("📝 偏見檢查評審中...")
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
                    result = {"raw_output": content, "bias_score": 5}
            except json.JSONDecodeError:
                result = {"raw_output": content, "bias_score": 5}

            return result

        except Exception as e:
            return {"error": str(e), "bias_score": 0}


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
