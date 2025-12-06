from config import config
import json
from typing import Dict, Any

from LM_AIG.ItemWriterAgent import ItemWritingAgent
from LM_AIG.CriticAgent import CriticAgent

class LM_AIG_System:
    """
    完整的 LM-AIG 系統，整合題目生成、評審和資料分析
    """

    def __init__(self):
        self.item_writer = ItemWritingAgent()
        self.critic = CriticAgent()

    def run_complete_workflow(self, specifications: str, num_items: int = 5,
                              max_iterations: int = 3) -> Dict[str, Any]:
        """
        執行完整的工作流程

        Args:
            specifications: 題目規格
            num_items: 題目數量
            max_iterations: 最大改進迭代次數

        Returns:

            {
                "original_specifications": str,
                "iterations": List[Dict],
                "final_items": List[str],
                "analysis_results": Dict[str, Any]
            }
        """
        workflow_results = {
            "original_specifications": specifications,
            "iterations": [],
            "final_items": None,
        }

        print(f"🚀 開始 LM-AIG 工作流程")
        print(f"📝 規格: {specifications}")
        print(f"🔢 目標題目數量: {num_items}")
        print("-" * 50)

        current_items = None
        previous_review = {}
        review_result = {}
        
        for iteration in range(max_iterations):
            print(f"\n🔄 第 {iteration + 1} 次迭代")

            # 第一次迭代：生成題目；後續迭代：改進題目
            if iteration == 0:
                print("📝 生成初始題目...")
                generation_result = self.item_writer.generate_items(
                    specifications, num_items)
            else:
                print("🔧 根據評審建議改進題目...")
                feedback = previous_review.get("recommendations", "請改進題目品質")
                generation_result = self.item_writer.refine_items(
                    current_items, str(feedback), specifications, num_items)

            if "error" in generation_result:
                print(f"❌ 生成錯誤: {generation_result['error']}")
                continue

            current_items = generation_result.get("items", [])
            print(f"✅ 已生成 {len(current_items)} 個題目")
            print(f"題目內容:")
            for i, item in enumerate(current_items, 1):
                print(f"{i:2}. {item.get('item') if isinstance(item, dict) else item}")

            # 評審題目
            print("🔍 評審題目品質...")
            review_result = self.critic.review_items(current_items,
                                                     specifications)

            iteration_result = {
                "iteration": iteration + 1,
                "generated_items": current_items,
                "generation_result": generation_result,
                "review_result": review_result,
                "overall_score": review_result.get("overall_score", 0)
            }

            # 顯示 reveiw result
            print(f"""🔍 評審結果: 
                  內容效度評估：{review_result.get("individual_reviews").get("content_review")}
                    語言學評估：{review_result.get("individual_reviews").get("linguistic_review")}
                    偏見檢查評估：{review_result.get("individual_reviews").get("bias_review")}
                    元評審結果：{review_result.get("meta_review")}""")

            workflow_results["iterations"].append(iteration_result)

            print(f"📊 綜合評分: {review_result.get('overall_score', 0)}/10")

            # 先檢查題目數，再檢查是否達到滿意標準
            if len(current_items) < num_items:
                print(f"❌ 題目數量不足 (需要 {num_items}，但只有 {len(current_items)})，繼續改進")
                previous_review = review_result.get("meta_review", {})

                previous_review = previous_review + f"\n\n 此外，請增加題目數量至至少 {num_items} 個。"
                continue
            elif review_result.get("overall_score", 0) >= 7:  # 7分以上算及格
                print("✅ 題目品質已達標準，結束迭代")
                break
            elif not review_result.get("meta_review", {}).get("regenerate_recommended", True):
                print("✅ 評審建議繼續使用當前版本")
                break
            else:
                print("⚠️ 需要繼續改進")
                previous_review = review_result.get("meta_review", {})

        workflow_results["final_items"] = current_items
        return workflow_results

    def display_results(self, results: Dict[str, Any]):
        """顯示工作流程結果"""
        print("\n" + "="*60)
        print("📋 LM-AIG 系統執行結果")
        print("="*60)

        print(f"\n📝 原始規格: {results['original_specifications']}")
        print(f"🔄 迭代次數: {len(results['iterations'])}")

        # 顯示最終題目
        final_items = results.get("final_items", [])

        # 如果是字串 JSON 格式，先解析為列表
        if isinstance(final_items, str):
            try:
                final_items = json.loads(final_items)
            except json.JSONDecodeError:
                final_items = [final_items]

        if final_items:
            print(f"\n📊 最終題目 ({len(final_items)} 個):")
        for i, item in enumerate(final_items, 1):
            print(f"{i:2}. {item}")

        # 顯示評審歷程
        print(f"\n📈 評分歷程:")
        for iteration in results["iterations"]:
            score = iteration.get("overall_score", 0)
            print(f"  第 {iteration['iteration']} 次迭代: {score}/10")