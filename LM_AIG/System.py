import datetime
from typing import Dict, Any

from LM_AIG.ItemWriterAgent import ItemWritingAgent
from LM_AIG.CriticAgent import CriticAgent
from LM_AIG.ConstructSettingAgent import ConstructSettingAgent


class LM_AIG_System:
    """
    完整的 LM-AIG 系統，整合題目生成、評審和資料分析
    """

    def __init__(self):
        self.item_writer = ItemWritingAgent()
        self.critic = CriticAgent()
        self.construct_agent = ConstructSettingAgent()

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
        print(f"🔢 目標題目數量: {num_items}")
        print("-" * 50)

        current_items = None
        previous_review = {}
        review_result = {}

        # 產生心理構念
        print("🧠 產生心理構念定義...")
        construct_definition = self.construct_agent.draft_construct_definition(
            specifications)
        print(f"心理構念定義：\n{construct_definition}\n")

        for iteration in range(max_iterations):
            print("="*40)
            print(f"🔄 第 {iteration + 1} 次迭代")

            # 第一次迭代：生成題目；後續迭代：改進題目
            if iteration == 0:
                print("📝 生成初始題目...")
                generation_result = self.item_writer.generate_items(
                    specifications, num_items, construct_definition)
            else:
                print("🔧 根據評審建議改進題目...")
                feedback = previous_review.get("recommendations", "請改進題目品質")
                generation_result = self.item_writer.refine_items(
                    current_items, str(feedback), specifications, num_items)

            if "error" in generation_result:
                print(f"❌ 生成錯誤: {generation_result['error']}")
                print(f"重新嘗試第 {iteration + 1} 次迭代...")
                iterartion -= 1 # 減少一次迭代計數，重新嘗試
                continue

            current_items = generation_result
            
            print(f"✅ 已生成 {len(current_items)} 個題目")
            print(f"題目內容:")
            for i, item in enumerate(current_items, 1):
                print(
                    f"{i:2}. {item.get('item') if isinstance(item, dict) else item}")

            print("-"*40)
            # 評審題目
            print("🔍 評審題目品質...")
            review_result = self.critic.review_items(
                items=current_items,
                specification=specifications,
                construct=construct_definition
            )

            iteration_result = {
                "iteration": iteration + 1,
                "generated_items": current_items,
                "generation_result": generation_result,
                "review_result": review_result,
                "overall_score": review_result.get("overall_score", 0)
            }

            # 顯示 reveiw result
            individual_reviews: dict = review_result.get("individual_reviews", {})
            print("🔍 評審結果: ")
            
            print("-"*40)
            print(f"內容效度評估：{individual_reviews.get("content_review", {}).get("validity_score", "無回傳評分")}")
            for idx, suggestion in enumerate(individual_reviews.get("content_review", {}).get("suggestions", []), 1):
                print(f"  建議 {idx}: {suggestion}")
            print("-"*40)
            print(f"語言學評估：{individual_reviews.get("linguistic_review", {}).get("linguistic_score", "無回傳評分")}")
            for idx, suggestion in enumerate(individual_reviews.get("linguistic_review", {}).get("suggestions", []), 1):
                print(f"  建議 {idx}: {suggestion}")
            print("-"*40)
            print(f"偏見檢查評估：{individual_reviews.get("bias_review", {}).get("bias_score", "無回傳評分")}")
            for idx, suggestion in enumerate(individual_reviews.get("bias_review", {}).get("suggestions", []), 1):
                print(f"  建議 {idx}: {suggestion}")
            print("-"*40)
            print(f"元評審結果：{review_result.get("meta_review").get("overall_score", "無回傳評分")}")
            for idx, recommendation in enumerate(review_result.get("meta_review").get("recommendations", []), 1):
                print(f"  建議 {idx}: {recommendation}")
            
            print("-"*40)
            workflow_results["iterations"].append(iteration_result)

            print(f"📊 綜合評分: {review_result.get('overall_score', 0)}/10")

            if len(current_items) < num_items:
                print(
                    f"⚠️ 生成的題目數量不足（{len(current_items)}/{num_items}），將重新生成題目。")
                previous_review=review_result.get("meta_review", {})
                previous_review["recommendations"].append(f"Currently only have generated {len(current_items)} items, please generate {num_items} items.")
            elif review_result.get("overall_score", 0) >= 8:  # 8分以上算及格
                print("✅ 題目品質已達標準，結束迭代")
                break
            elif not review_result.get("meta_review", {}).get("regenerate_recommended", True):
                print("✅ 評審建議繼續使用當前版本")
                break
            else:
                print("⚠️ 需要繼續改進")
                previous_review=review_result.get("meta_review", {})

        workflow_results["final_items"]=current_items

        self.display_results(workflow_results, specifications)

        return workflow_results

    def display_results(self, results: Dict[str, Any], specifications: str):
        """顯示工作流程結果"""
        # 顯示最終結果
        print("\n" + "="*40)
        print("🚀 LM-AIG 工作流程完成！")
        print("="*40)

        print("\n📝 最終題目列表:")
        for idx, item in enumerate(results.get("final_items", []), 1):
            print(f"  {idx}. {item['item']} (Construct: {item['psychological_construct']})")

        print("\n🔄 各迭代評分:")
        for iteration in results.get("iterations", []):
            score = iteration.get("overall_score", 0)
            print(f"  第 {iteration['iteration']} 次迭代: {score}/10")

        # 將結果儲存為文字檔
        filename = f"lm_aig_workflow_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open("AIG_results/"+filename, "w", encoding="utf-8") as f:
            f.write("LM-AIG 工作流程結果\n")
            f.write("="*40 + "\n\n")
            f.write("題目生成日期: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
            f.write(f"使用者輸入規格:\n{specifications}\n\n")
            f.write("題目列表:\n")
            
            # 根據構念分類題目
            constructs = []  # A list of constructs
            items_by_construct = {} # A dictionary to hold items by construct
            for item in results.get("final_items", []):
                construct = item.get("psychological_construct", "未分類")
                if construct not in constructs:
                    constructs.append(construct)
                    items_by_construct[construct] = []
                items_by_construct[construct].append(item.get("item", ""))

            for construct in constructs:
                f.write(f"\n構念: {construct}\n")
                for idx, item in enumerate(items_by_construct[construct], 1):
                    f.write(f"  {idx}. {item}\n")

            f.write("\n各迭代評分:\n")
            for iteration in results["iterations"]:
                score = iteration.get("overall_score", 0)
                f.write(f"第 {iteration['iteration']} 次迭代: {score}/10\n")

        print(f"\n✅ 結果已儲存至 {filename}")