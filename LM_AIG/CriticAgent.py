from config import config
from typing import List, Dict, Any

from LM_AIG.CriticAgents.ContentReviewer import ContentReviewer
from LM_AIG.CriticAgents.LinguisticReviewer import LinguisticReviewer
from LM_AIG.CriticAgents.BiasReviewer import BiasReviewer
from LM_AIG.CriticAgents.MetaReviewer import MetaReviewer

class CriticAgent:
    """
    評審代理人，包含多個子評審員來評估測驗題目的品質
    """

    def __init__(self, model: str = None):
        self.model = model or config.model_name
        self.content_reviewer = ContentReviewer(self.model)
        self.linguistic_reviewer = LinguisticReviewer(self.model)
        self.bias_reviewer = BiasReviewer(self.model)
        self.meta_reviewer = MetaReviewer(self.model)

    def review_items(self, items: List[str], specification: str = "", construct: str = "") -> Dict[str, Any]:
        """
        全面評審測驗題目

        Args:
            items: 題目列表
            specification: 題目規格說明
            construct: 心理建構名稱

        Returns:
            評審結果
        """

        # 各子評審員的評審結果
        reviews = {
            "content_review": self.content_reviewer.review(items, construct),
            "linguistic_review": self.linguistic_reviewer.review(items),
            "bias_review": self.bias_reviewer.review(items),
        }

        # 元評審員整合所有評審結果
        meta_review = self.meta_reviewer.integrate_reviews(
            reviews=reviews,
            items=items,
            specifications=specification,
            )

        return {
            "individual_reviews": reviews,
            "meta_review": meta_review,
            "overall_score": meta_review.get("overall_score", 0),
            "recommendations": meta_review.get("recommendations", [])
        }
