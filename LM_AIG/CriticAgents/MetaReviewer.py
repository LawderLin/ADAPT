import ollama
import json
import re
from typing import List, Dict, Any
from pydantic import BaseModel

from LM_AIG.JSONFormatAgent import JSONFormatAgent

class MetaReviewer:
    """元評審員，整合所有評審結果"""

    def __init__(self, model: str):
        self.model = model
        self.system_prompt = self._default_system_prompt()
        self.schema = self._define_schema()

    def _default_system_prompt(self) -> str:
        """預設的系統提示詞"""
        return """You are a senior expert in test item review. Please integrate the following review results from different reviewers and provide an overall evaluation and suggestions for improvement. Also, check if there is any minor issues missed by other reviewers, such as item
        
        Given the original items and the review results from content validity, linguistic quality, and bias check reviewers, please provide:

        Output in JSON format, including:
        - overall_score: Overall score (1-10)
        - strengths: A String describing the main strengths
        - major_issues: A String describing the major issues
        - recommendations: A list of Strings describing recommendations for improvement
        - regenerate_recommended: Whether regeneration is recommended (boolean)

        Your output should ONLY be the JSON format as specified above. Do not include any other content outside the JSON format.
        """
    
    def _define_schema(self) -> BaseModel:
        """定義輸出 JSON 結構"""
        class ReviewSchema(BaseModel):
            overall_score: int
            strengths: str
            major_issues: str
            recommendations: list[str]
            regenerate_recommended: bool

        return ReviewSchema


    def integrate_reviews(self, reviews: Dict[str, Any], items: List[str], specifications: str) -> Dict[str, Any]:
        """整合所有評審結果"""
        content_review = reviews.get("content_review", {})
        linguistic_review = reviews.get("linguistic_review", {})
        bias_review = reviews.get("bias_review", {})

        prompt = f"""
Specifications for the psychological test items:
{specifications}

Original items:
{'\n'.join([f"- {item['item']} ({item['psychological_construct']})" for item in items])}

Review results from each reviewer:
Content validity review: 
- validity_score: {content_review.get('validity_score', 'N/A')}
{'\n'.join([f"- {s}" for s in content_review.get('suggestions', [])])}
Linguistic review: 
- linguistic_score: {linguistic_review.get('linguistic_score', 'N/A')}
{'\n'.join([f"- {s}" for s in linguistic_review.get('suggestions', [])])}
Bias check review: 
- bias_score: {bias_review.get('bias_score', 'N/A')}
{'\n'.join([f"- {s}" for s in bias_review.get('suggestions', [])])}
    """
        return self._get_review_response(prompt)

    def _get_review_response(self, prompt: str) -> Dict[str, Any]:
        """獲取評審回應"""
        print("整合所有評審結果...")
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": self.system_prompt}, {"role": "user", "content": prompt}],
                options={"temperature": 0.3},
                format=self.schema.model_json_schema()
            )

            content = response['message']['content']
            result = JSONFormatAgent().format_to_json(content)
            return result[0]

        except Exception as e:
            print("Failed to parse reviews. Retrying...")
            return self._get_review_response(prompt)
